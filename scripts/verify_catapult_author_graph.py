#!/usr/bin/env python3
"""Verify true author DiskANN graph/payload format, used by original Catapult."""
import struct,sys
from pathlib import Path
def main():
    g,p=map(Path,sys.argv[1:])
    with p.open("rb") as f: n,d=struct.unpack("<II",f.read(8))
    assert (n,d)==(31950,64),(n,d)
    with g.open("rb") as f: file_size,degree,entry,frozen=struct.unpack("<QIIQ",f.read(24))
    assert file_size==g.stat().st_size,(file_size,g.stat().st_size)
    assert 10<=degree<=1000 and entry<31950
    assert frozen==0
    print("AUTHOR_DISKANN_MEMORY_GRAPH_VALIDATED",n,d,degree,entry,file_size,flush=True)
if __name__=="__main__":main()
