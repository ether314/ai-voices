#!/bin/bash
set -euo pipefail
export TOKEN="${HF_TOKEN:-}"
export BLOB=/cache/huggingface/hub/models--ResembleAI--chatterbox-nano/blobs
export SNAP=/cache/huggingface/hub/models--ResembleAI--chatterbox-nano/snapshots/71ccd1d0081b430592cea481f4307e764e07bc64
mkdir -p "$BLOB" "$SNAP"

python3 - <<'PY'
import glob, hashlib, os, shutil, time, urllib.request

BLOB = os.environ["BLOB"]
SNAP = os.environ["SNAP"]
TOKEN = os.environ.get("TOKEN", "")

# filename -> expected LFS sha256 + size
FILES = {
    "s3gen.safetensors": (
        "2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e",
        1056484620,
    ),
    "s3gen_meanflow.safetensors": (
        "d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f",
        1064875036,
    ),
}

# Keep largest incomplete per oid
by = {}
for p in glob.glob(os.path.join(BLOB, "*.incomplete")):
    base = os.path.basename(p).split(".")[0]
    by.setdefault(base, []).append(p)
for base, paths in by.items():
    paths.sort(key=os.path.getsize, reverse=True)
    keep = paths[0]
    for p in paths[1:]:
        print("remove", p, os.path.getsize(p), flush=True)
        os.remove(p)
    print("keep", keep, os.path.getsize(keep), flush=True)

# Seed named partials from matching oid incompletes
for name, (oid, _size) in FILES.items():
    dst = os.path.join(BLOB, f"{name}.partial")
    final = os.path.join(BLOB, oid)
    if os.path.exists(final) and os.path.getsize(final) == _size:
        print("already complete", name, flush=True)
        link = os.path.join(SNAP, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(f"../../blobs/{oid}", link)
        continue
    srcs = [
        p
        for p in glob.glob(os.path.join(BLOB, f"{oid}.*.incomplete"))
        if os.path.getsize(p) > 0
    ]
    srcs.sort(key=os.path.getsize, reverse=True)
    # Always reseed from the matching oid incomplete (prior runs may have
    # mixed wrong file content into *.partial).
    if srcs:
        src = srcs[0]
        shutil.copy2(src, dst)
        print("seeded", name, "from", os.path.basename(src), os.path.getsize(src), flush=True)
    else:
        open(dst, "wb").close()
        print("start fresh", name, flush=True)


def download(name: str, oid: str, expected: int) -> None:
    final = os.path.join(BLOB, oid)
    if os.path.exists(final) and os.path.getsize(final) == expected:
        print(f"SKIP {name} already have {oid}", flush=True)
        link = os.path.join(SNAP, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(f"../../blobs/{oid}", link)
        return

    url = f"https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/{name}"
    tmp = os.path.join(BLOB, f"{name}.partial")
    if not os.path.exists(tmp):
        open(tmp, "wb").close()
    have = os.path.getsize(tmp)
    print(f"RESUMING {name} (have {have} / {expected})", flush=True)
    headers = {"User-Agent": "chatterbox-resume/1.0"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    retries = 0
    while os.path.getsize(tmp) < expected:
        have = os.path.getsize(tmp)
        req_headers = dict(headers)
        if have > 0:
            req_headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                code = getattr(resp, "status", None) or resp.getcode()
                if code == 200 and have > 0:
                    print(
                        f"WARN server returned 200 with have={have}; truncating and restarting",
                        flush=True,
                    )
                    open(tmp, "wb").close()
                    continue
                mode = "ab" if have > 0 and code == 206 else ("wb" if code == 200 else "ab")
                last = time.time()
                with open(tmp, mode) as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        now = time.time()
                        if now - last >= 5:
                            print(f"  {name}: {os.path.getsize(tmp)} / {expected}", flush=True)
                            last = now
            retries = 0
        except Exception as e:
            retries += 1
            print(f"retry {retries}: {e}", flush=True)
            if retries > 40:
                raise
            time.sleep(min(2 * retries, 30))
    got = os.path.getsize(tmp)
    print(f"GOT {name} size={got} expected={expected}", flush=True)
    if got != expected:
        raise SystemExit(f"SIZE MISMATCH for {name}")
    print(f"hashing {name}...", flush=True)
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    if digest != oid:
        raise SystemExit(f"HASH MISMATCH for {name}: got {digest} expected {oid}")
    os.replace(tmp, final)
    link = os.path.join(SNAP, name)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(f"../../blobs/{oid}", link)
    print(f"OK {name} -> {oid}", flush=True)


for name, (oid, size) in FILES.items():
    download(name, oid, size)
print("ALL_DONE", flush=True)
PY
