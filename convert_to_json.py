# convert_tuple_lines_to_jsonl.py
from ast import literal_eval

import orjson as _json

dumps = lambda obj: _json.dumps(obj).decode()
loads = _json.loads

IN_PATH  = "branch_pairs.txt"
OUT_PATH = "branch_pairs.jsonl"
CHUNK    = 200_000

def convert():
    n = 0
    with open(IN_PATH, "r", encoding="utf-8", buffering=1024*1024) as fin, \
         open(OUT_PATH, "w", encoding="utf-8", buffering=1024*1024) as fout:
        buf = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            x, a, b, c = literal_eval(line)  # safe parse
            buf.append((x, a, b, c))         # write as a JSON array
            if len(buf) >= CHUNK:
                for row in buf:
                    fout.write(dumps(row) + "\n")
                n += len(buf); buf.clear()
        if buf:
            for row in buf:
                fout.write(dumps(row) + "\n")
            n += len(buf)
    print(f"wrote {n:,} rows to {OUT_PATH}")

if __name__ == "__main__":
    convert()
