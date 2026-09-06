"""Scan bundled DLLs for dependencies on a needle (who pulls ICU chain in)."""
import sys
from pathlib import Path

dist, needle = Path(sys.argv[1]), sys.argv[2].encode().lower()
hits = [(p.name, p.stat().st_size) for p in dist.rglob("*.dll") if needle in p.read_bytes().lower()]
for name, size in hits:
    print(f"{name}: {size / 2**20:.1f} MB")
print(f"{len(hits)} binaries reference '{sys.argv[2]}'")

