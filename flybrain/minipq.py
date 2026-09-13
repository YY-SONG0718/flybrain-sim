"""A minimal Parquet reader: numeric columns, PLAIN or dictionary encoding,
uncompressed or Brotli. Enough for the FlyWire connectivity dump, so that the
project does not depend on pyarrow.

Decompression uses the system libbrotli through ctypes.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import struct
from pathlib import Path

import numpy as np
from loguru import logger

from .tcompact import ThriftStruct, read_struct, read_varint

# Parquet enums (only the values this reader needs)
CODEC_UNCOMPRESSED = 0
CODEC_BROTLI = 4
PAGE_DICTIONARY = 2
ENCODING_PLAIN = 0
ENCODING_PLAIN_DICTIONARY = 2
ENCODING_RLE_DICTIONARY = 8

# FileMetaData / ColumnMetaData / PageHeader field ids from parquet.thrift
FMD_SCHEMA, FMD_NUM_ROWS, FMD_ROW_GROUPS = 2, 3, 4
RG_COLUMNS = 1
CC_META = 3
CM_TYPE, CM_PATH, CM_CODEC, CM_NUM_VALUES = 1, 3, 4, 5
CM_TOTAL_COMPRESSED, CM_DATA_PAGE_OFFSET, CM_DICT_PAGE_OFFSET = 7, 9, 11
PH_TYPE, PH_UNCOMPRESSED_SIZE, PH_COMPRESSED_SIZE, PH_DATA_HEADER = 1, 2, 3, 5
DPH_NUM_VALUES, DPH_ENCODING = 1, 2
SCHEMA_NAME = 4

PHYSICAL_TYPE_TO_NUMPY = {1: np.int32, 2: np.int64, 4: np.float32, 5: np.float64}

BROTLI_LIBRARY_NAMES = ("libbrotlidec.so.1", "libbrotlidec.dylib", "libbrotlidec.1.dylib", "brotlidec")
_brotli_library: ctypes.CDLL | None = None


def load_brotli_library() -> ctypes.CDLL:
    """
    Load the system Brotli decoder once, under its Linux or macOS name.

    :return: The loaded library with ``BrotliDecoderDecompress`` typed.
    :raises ImportError: if no Brotli library is installed.
    """
    global _brotli_library
    if _brotli_library is not None:
        return _brotli_library
    candidates = list(BROTLI_LIBRARY_NAMES)
    found = ctypes.util.find_library("brotlidec")
    if found:
        candidates.insert(0, found)
    for name in candidates:
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        library.BrotliDecoderDecompress.restype = ctypes.c_int
        library.BrotliDecoderDecompress.argtypes = [
            ctypes.c_size_t, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p]
        _brotli_library = library
        logger.debug("brotli decoder loaded from {}", name)
        return library
    raise ImportError(
        "No Brotli library found. Install pyarrow (`uv pip install pyarrow`) and the "
        "connectome loader will use it instead; or install brotli (`brew install brotli`).")


def brotli_decompress(compressed: bytes, uncompressed_size: int) -> bytes:
    """
    Decompress a Brotli page body with the system Brotli library.

    :param compressed: Compressed bytes.
    :param uncompressed_size: Expected size of the output, from the page header.
    :return: Decompressed bytes.
    :raises RuntimeError: if the library reports a decode failure.
    """
    library = load_brotli_library()
    output = ctypes.create_string_buffer(uncompressed_size)
    output_size = ctypes.c_size_t(uncompressed_size)
    status = library.BrotliDecoderDecompress(len(compressed), compressed,
                                             ctypes.byref(output_size), output)
    if status != 1:
        raise RuntimeError("brotli decode failed")
    return output.raw[:output_size.value]


def unpack_bits(data: bytes, bit_width: int, count: int) -> np.ndarray:
    """
    Decode LSB-first bit-packed unsigned integers.

    :param data: Packed bytes.
    :param bit_width: Bits per value.
    :param count: Number of values to decode.
    :return: int64 array of length ``count``.
    """
    if bit_width == 0:
        return np.zeros(count, dtype=np.int64)
    needed_bytes = (count * bit_width + 7) // 8
    padded = data[:needed_bytes] + b"\x00" * (needed_bytes + 8 - len(data[:needed_bytes]))
    bits = np.unpackbits(np.frombuffer(padded, dtype=np.uint8), bitorder="little")
    bits = bits[:count * bit_width].reshape(count, bit_width)
    weights = (1 << np.arange(bit_width)).astype(np.int64)
    return bits.astype(np.int64) @ weights


def decode_rle_bitpacked_hybrid(data: bytes, bit_width: int, count: int) -> np.ndarray:
    """
    Decode Parquet's RLE / bit-packing hybrid encoding.

    :param data: Encoded bytes.
    :param bit_width: Bits per value.
    :param count: Number of values to decode.
    :return: int64 array of length ``count``.
    """
    values = np.empty(count, dtype=np.int64)
    filled = 0
    position = 0
    byte_width = (bit_width + 7) // 8
    while filled < count:
        header, position = read_varint(data, position)
        if header & 1:                                  # bit-packed run
            group_count = header >> 1
            run_length = group_count * 8
            run_bytes = group_count * bit_width
            run = unpack_bits(data[position:position + run_bytes], bit_width, run_length)
            position += run_bytes
        else:                                           # repeated-value run
            run_length = header >> 1
            raw = data[position:position + byte_width] + b"\x00" * (8 - byte_width)
            run = np.full(run_length, struct.unpack("<Q", raw)[0], dtype=np.int64)
            position += byte_width
        take = min(run_length, count - filled)
        values[filled:filled + take] = run[:take]
        filled += take
    return values


class ParquetFile:
    """Read whole columns from a Parquet file."""

    def __init__(self, path: str | Path) -> None:
        """
        Open a Parquet file and parse its footer.

        :param path: Path to the .parquet file.
        """
        self.path = Path(path)
        self._file = open(self.path, "rb")
        self._file.seek(-8, 2)
        footer_length = struct.unpack("<I", self._file.read(4))[0]
        self._file.seek(-8 - footer_length, 2)
        self.metadata, _ = read_struct(self._file.read(footer_length), 0)
        self.num_rows: int = self.metadata[FMD_NUM_ROWS]
        self.columns: list[str] = [
            element[SCHEMA_NAME].decode() for element in self.metadata[FMD_SCHEMA][1:]]
        logger.debug("opened {} : {} rows, columns {}", self.path.name, self.num_rows, self.columns)

    def read_column(self, name: str) -> np.ndarray:
        """
        Read one whole column across all row groups.

        :param name: Column name as in the schema.
        :return: 1-D array of the column's values.
        :raises KeyError: if no column has that name.
        """
        chunks = []
        for row_group in self.metadata[FMD_ROW_GROUPS]:
            for column_chunk in row_group[RG_COLUMNS]:
                column_meta = column_chunk[CC_META]
                if column_meta[CM_PATH][0].decode() == name:
                    chunks.append(self._read_column_chunk(column_meta))
        if not chunks:
            raise KeyError(f"no column named {name!r}; have {self.columns}")
        return np.concatenate(chunks)

    def _read_column_chunk(self, column_meta: ThriftStruct) -> np.ndarray:
        """
        Read and decode one column chunk.

        :param column_meta: The chunk's ColumnMetaData struct.
        :return: Decoded values for the chunk.
        """
        dtype = PHYSICAL_TYPE_TO_NUMPY[column_meta[CM_TYPE]]
        codec = column_meta[CM_CODEC]
        expected_values = column_meta[CM_NUM_VALUES]
        start = column_meta.get(CM_DICT_PAGE_OFFSET) or column_meta[CM_DATA_PAGE_OFFSET]
        self._file.seek(start)
        blob = self._file.read(column_meta[CM_TOTAL_COMPRESSED])

        position = 0
        dictionary: np.ndarray | None = None
        pages: list[np.ndarray] = []
        values_read = 0
        while values_read < expected_values and position < len(blob):
            page_header, body_start = read_struct(blob, position)
            body = blob[body_start:body_start + page_header[PH_COMPRESSED_SIZE]]
            position = body_start + page_header[PH_COMPRESSED_SIZE]
            if codec == CODEC_BROTLI:
                body = brotli_decompress(body, page_header[PH_UNCOMPRESSED_SIZE])
            elif codec != CODEC_UNCOMPRESSED:
                raise NotImplementedError(f"compression codec {codec} not supported")

            if page_header[PH_TYPE] == PAGE_DICTIONARY:
                dictionary = np.frombuffer(body, dtype=dtype)
                continue

            data_header = page_header[PH_DATA_HEADER]
            page_values = data_header[DPH_NUM_VALUES]
            encoding = data_header[DPH_ENCODING]
            pages.append(self._decode_data_page(body, dtype, encoding, page_values, dictionary))
            values_read += page_values
        return np.concatenate(pages)

    @staticmethod
    def _decode_data_page(body: bytes, dtype: type, encoding: int, page_values: int,
                          dictionary: np.ndarray | None) -> np.ndarray:
        # optional column: definition levels come first, RLE-encoded with bit width 1
        """
        Decode one data page of an optional numeric column.

        :param body: Uncompressed page body.
        :param dtype: NumPy dtype of the column.
        :param encoding: Parquet encoding code.
        :param page_values: Number of values (including nulls) in the page.
        :param dictionary: Dictionary page values, if the page is dictionary-encoded.
        :return: Array of length ``page_values`` with nulls as NaN or 0.
        """
        definition_length = struct.unpack("<I", body[:4])[0]
        definition_levels = decode_rle_bitpacked_hybrid(body[4:4 + definition_length], 1, page_values)
        offset = 4 + definition_length
        present_count = int(definition_levels.sum())

        if encoding in (ENCODING_PLAIN_DICTIONARY, ENCODING_RLE_DICTIONARY):
            if dictionary is None:
                raise ValueError("dictionary-encoded page without a dictionary page")
            bit_width = body[offset]
            indices = decode_rle_bitpacked_hybrid(body[offset + 1:], bit_width, present_count)
            values = dictionary[indices]
        elif encoding == ENCODING_PLAIN:
            values = np.frombuffer(body[offset:], dtype=dtype, count=present_count)
        else:
            raise NotImplementedError(f"encoding {encoding} not supported")

        if present_count == page_values:
            return values.astype(dtype)
        fill = np.nan if dtype == np.float64 else 0
        full = np.full(page_values, fill, dtype=dtype)
        full[definition_levels.astype(bool)] = values
        return full
