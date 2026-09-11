"""Minimal Parquet reader: INT64/DOUBLE columns, PLAIN + RLE_DICTIONARY, ZSTD/SNAPPY-free.
Enough to read the FlyWire connectivity dump without pyarrow."""
import importlib, struct, sys, ctypes, ctypes.util, numpy as np
try:
    from .tcompact import read_struct
except ImportError:
    from tcompact import read_struct

_CDLL_NAMES = {
    "darwin": ["libbrotlidec.dylib", "libbrotlidec.1.dylib",
               "/opt/homebrew/lib/libbrotlidec.dylib",
               "/usr/local/lib/libbrotlidec.dylib"],
    "win32": ["brotlidec.dll", "libbrotlidec.dll"],
}
_CDLL_DEFAULT = ["libbrotlidec.so.1", "libbrotlidec.so"]

_brotli_decompress = None


def _load_ctypes_backend():
    """Find the brotli decoder shared library. Return a decompress function."""
    names = list(_CDLL_NAMES.get(sys.platform, _CDLL_DEFAULT))
    found = ctypes.util.find_library("brotlidec")
    if found:
        names.insert(0, found)
    lib = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return None
    lib.BrotliDecoderDecompress.restype = ctypes.c_int
    lib.BrotliDecoderDecompress.argtypes = [ctypes.c_size_t, ctypes.c_char_p,
                                            ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.c_char_p]

    def decompress(buf, outsize):
        out = ctypes.create_string_buffer(max(outsize, 1))
        n = ctypes.c_size_t(outsize)
        if lib.BrotliDecoderDecompress(len(buf), buf, ctypes.byref(n), out) != 1:
            raise RuntimeError("brotli decode failed")
        return out.raw[:n.value]

    return decompress


def _load_backend():
    """Return a decompress(buf, outsize) -> bytes function, or raise."""
    for modname in ("brotli", "brotlicffi"):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        return lambda buf, outsize, _d=mod.decompress: _d(bytes(buf))
    backend = _load_ctypes_backend()
    if backend is not None:
        return backend
    raise RuntimeError(
        "This Parquet file uses BROTLI compression, and no brotli decoder was found.\n"
        "Install the Python package:   pip install brotli\n"
        "Or install the system library: brew install brotli  (macOS), "
        "apt install libbrotli1  (Debian/Ubuntu)"
    )


def brotli_decompress(buf, outsize):
    global _brotli_decompress
    if _brotli_decompress is None:
        _brotli_decompress = _load_backend()
    return _brotli_decompress(buf, outsize)

def _unpack_bits(data, width, count):
    """LSB-first bit-packed unsigned ints."""
    if width == 0:
        return np.zeros(count, dtype=np.int64)
    nbytes = (count * width + 7) // 8
    buf = data[:nbytes] + b"\x00" * (nbytes + 8 - len(data[:nbytes]))
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    need = count * width
    bits = bits[:need].reshape(count, width)
    weights = (1 << np.arange(width)).astype(np.int64)
    return bits.astype(np.int64) @ weights

def rle_hybrid(data, width, count):
    """Decode RLE/bit-packing hybrid run into `count` values."""
    out = np.empty(count, dtype=np.int64)
    filled = 0
    p = 0
    bytewidth = (width + 7) // 8
    while filled < count:
        header, p = _varint(data, p)
        if header & 1:                      # bit-packed
            groups = header >> 1
            n = groups * 8
            nbytes = groups * width
            vals = _unpack_bits(data[p:p + nbytes], width, n)
            p += nbytes
            take = min(n, count - filled)
            out[filled:filled + take] = vals[:take]
            filled += take
        else:                               # RLE run
            run = header >> 1
            raw = data[p:p + bytewidth] + b"\x00" * (8 - bytewidth)
            val = struct.unpack("<Q", raw)[0]
            p += bytewidth
            take = min(run, count - filled)
            out[filled:filled + take] = val
            filled += take
    return out

def _varint(b, p):
    r = 0; s = 0
    while True:
        x = b[p]; p += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, p
        s += 7

_NP = {1: np.int32, 2: np.int64, 4: np.float32, 5: np.float64}

class ParquetFile:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.f.seek(-8, 2)
        flen = struct.unpack("<I", self.f.read(4))[0]
        self.f.seek(-8 - flen, 2)
        self.meta, _ = read_struct(self.f.read(flen), 0)
        self.num_rows = self.meta[3]
        self.columns = [e[4].decode() for e in self.meta[2][1:]]

    def read_column(self, name):
        chunks = []
        for rg in self.meta[4]:
            for ch in rg[1]:
                m = ch[3]
                if m[3][0].decode() != name:
                    continue
                chunks.append(self._read_chunk(m))
        return np.concatenate(chunks)

    def _read_chunk(self, m):
        ptype = m[1]
        dtype = _NP[ptype]
        codec = m[4]
        nvals = m[5]
        start = m.get(11) or m[9]           # dictionary page offset if present
        end = start + m[7]
        self.f.seek(start)
        blob = self.f.read(end - start)
        p = 0
        dictionary = None
        out = []
        got = 0
        while got < nvals and p < len(blob):
            hdr, np_ = read_struct(blob, p)
            body = blob[np_:np_ + hdr[3]]
            p = np_ + hdr[3]
            if codec == 4:
                body = brotli_decompress(body, hdr[2])
            elif codec != 0:
                raise NotImplementedError("codec %d" % codec)
            if hdr[1] == 2:                 # dictionary page
                dictionary = np.frombuffer(body, dtype=dtype)
                continue
            dph = hdr[5]
            n = dph[1]
            enc = dph[2]
            q = 0
            # optional column -> definition levels, RLE, bit width 1
            dlen = struct.unpack("<I", body[q:q + 4])[0]
            defs = rle_hybrid(body[q + 4:q + 4 + dlen], 1, n)
            q += 4 + dlen
            present = int(defs.sum())
            if enc in (2, 8):               # (PLAIN_)DICTIONARY / RLE_DICTIONARY
                width = body[q]; q += 1
                idx = rle_hybrid(body[q:], width, present)
                vals = dictionary[idx]
            elif enc == 0:                  # PLAIN
                vals = np.frombuffer(body[q:], dtype=dtype, count=present)
            else:
                raise NotImplementedError("encoding %d" % enc)
            if present == n:
                out.append(vals.astype(dtype))
            else:
                full = np.full(n, np.nan if dtype == np.float64 else 0, dtype=dtype)
                full[defs.astype(bool)] = vals
                out.append(full)
            got += n
        return np.concatenate(out)

    def to_dict(self, names=None):
        return {c: self.read_column(c) for c in (names or self.columns)}
