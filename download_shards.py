#!/usr/bin/env python3
"""Multi-threaded range-shard downloader for HF mirror (xet-bridge compatible).

Usage: download_shards.py <url> <output> [n_threads] [chunk_mb]
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "python-urllib/3.10"


def get_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0))


def fetch_shard(url, start, end, dest, idx, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            with open(dest, "r+b") as f:
                f.seek(start)
                f.write(data)
            return idx, start, len(data)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def main():
    url, out = sys.argv[1], sys.argv[2]
    n_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    chunk_mb = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    total = get_size(url)
    print(f"total={total / 1e9:.2f} GB, threads={n_threads}, "
          f"chunk={chunk_mb}MB", flush=True)
    with open(out, "wb") as f:
        f.truncate(total)
    chunk = chunk_mb * 1024 * 1024
    ranges = [(i, min(i + chunk - 1, total - 1)) for i in range(0, total, chunk)]
    t0 = time.time()
    done_bytes = [0]

    def report(fut):
        _, start, size = fut.result()
        done_bytes[0] += size
        elapsed = time.time() - t0
        print(f"  {done_bytes[0] / 1e9:.2f}/{total / 1e9:.2f} GB "
              f"({done_bytes[0] / total * 100:.1f}%) "
              f"{done_bytes[0] / elapsed / 1e6:.2f} MB/s", flush=True)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(fetch_shard, url, s, e, out, i)
                   for i, (s, e) in enumerate(ranges)]
        for fut in futures:
            report(fut)
    print(f"DONE {out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
