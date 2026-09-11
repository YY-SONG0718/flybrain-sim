import struct

def read_varint(b,p):
    r=0;s=0
    while True:
        x=b[p]; p+=1
        r |= (x & 0x7f) << s
        if not (x & 0x80): break
        s+=7
    return r,p

def zigzag(n): return (n>>1) ^ -(n&1)

def read_struct(b,p):
    out={}; last=0
    while True:
        h=b[p]; p+=1
        if h==0: break
        delta=(h>>4)&0xf; t=h&0xf
        if delta==0:
            zz,p=read_varint(b,p); fid=zigzag(zz)
        else:
            fid=last+delta
        last=fid
        if t==1: out[fid]=True; continue
        if t==2: out[fid]=False; continue
        v,p=read_val(b,p,t)
        out[fid]=v
    return out,p

def read_val(b,p,t):
    if t==1: return True,p
    if t==2: return False,p
    if t==3: return struct.unpack('b',b[p:p+1])[0],p+1
    if t in (4,5,6):
        zz,p=read_varint(b,p); return zigzag(zz),p
    if t==7: return struct.unpack('<d',b[p:p+8])[0],p+8
    if t==8:
        n,p=read_varint(b,p); return b[p:p+n],p+n
    if t in (9,10):
        h=b[p]; p+=1
        sz=(h>>4)&0xf; et=h&0xf
        if sz==15: sz,p=read_varint(b,p)
        lst=[]
        for _ in range(sz):
            v,p=read_val(b,p,et); lst.append(v)
        return lst,p
    if t==11:  # MAP
        sz,p=read_varint(b,p)
        if sz==0: return {},p
        kv=b[p]; p+=1
        kt=(kv>>4)&0xf; vt=kv&0xf
        d={}
        for _ in range(sz):
            k,p=read_val(b,p,kt); v,p=read_val(b,p,vt); d[k]=v
        return d,p
    if t==12:
        return read_struct(b,p)
    raise ValueError('bad type %d at %d'%(t,p))
