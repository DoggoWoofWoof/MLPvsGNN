"""Show the bytes around the first occurrence of a key name inside a zip member (no unpickling).

Run as: python -I stark_find.py <zip> <member> <key> [window]
Streams the member in 32 MB chunks, finds the first SHORT_BINUNICODE-encoded key (opcode byte 0x8c, a length byte,
then the key's text) and prints the window before and after it with unprintable bytes shown as '|'.
Nothing is written and nothing is executed.
"""
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
zp, member, key = sys.argv[1], sys.argv[2], sys.argv[3]
win = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
pat = b"\x8c" + bytes([len(key)]) + key.encode()
tail = b""
off = 0
CH = 32 * 1024 * 1024
with zipfile.ZipFile(zp) as zf:
    with zf.open(member) as f:
        while True:
            buf = f.read(CH)
            if not buf:
                print("not found in", off, "bytes")
                break
            data = tail + buf
            i = data.find(pat)
            if i >= 0:
                a = max(0, i - win)
                seg = data[a:i + win]
                txt = "".join(chr(c) if 32 <= c < 127 or c >= 160 else "|" for c in seg.decode("latin-1").encode("latin-1"))
                try:
                    txt = seg.decode("utf-8", "replace")
                    txt = "".join(ch if (ch.isprintable() and ch != "�") else "|" for ch in txt)
                except Exception:  # noqa: BLE001
                    pass
                print("found at byte", off - len(tail) + i)
                print(txt)
                break
            tail = data[-(win + 64):]
            off += len(buf)
