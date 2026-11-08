#!/usr/bin/env python3
"""Reproducible, lossless DedupArc extreme compression benchmark (stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tarfile
import time

SEED = 20261108

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def corpus(root, category, scale):
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    if category == "versioned":
        base = rng.randbytes(4*1024*1024*scale)
        for n in range(10):
            version = bytearray(base)
            if n:
                # Distinct point edits in an otherwise compressed/uncompressible binary.
                for _ in range(150*scale):
                    at = rng.randrange(len(version))
                    version[at] ^= rng.randint(1, 255)
            (root / f"build-{n:02}.bin").write_bytes(version)
    elif category == "random":
        for i in range(4*scale):
            (root / f"noise-{i:02}.bin").write_bytes(rng.randbytes(1024*1024))
    elif category == "text":
        template = '{"version": 42, "feature": "deduparc", "data": "shared templates", "enabled": true}\n'
        for i in range(1200*scale):
            text = (template + json.dumps({"id":i % 201,"variant":i%7}, sort_keys=True) + "\n") * (10+(i%9))
            (root / f"config-{i:05}.json").write_text(text, encoding="utf-8")
    return sorted(p for p in root.rglob("*") if p.is_file())

def run(args, stdout=None):
    began = time.perf_counter()
    subprocess.run(args, check=True, stdout=stdout or subprocess.DEVNULL,
                   stderr=subprocess.PIPE)
    return round(time.perf_counter()-began, 3)

def benchmark(binary, directory, category, scale):
    source = directory / category
    files = corpus(source, category, scale)
    raw = sum(p.stat().st_size for p in files)
    expected = {p.relative_to(source).as_posix():digest(p) for p in files}
    report = {"category":category,"files":len(files),"source_bytes":raw,
              "codecs":{},"sha256_roundtrip":"PASS"}
    for mode in ("maximum","extreme"):
        dest = directory / f"{category}-{mode}.darc"
        took = run([str(binary),"create","--"+mode,str(dest),str(source)])
        run([str(binary),"verify",str(dest)])
        output = directory / f"restore-{category}-{mode}"
        run([str(binary),"extract",str(dest),str(output)])
        recovered = output / category
        if {p.relative_to(recovered).as_posix():digest(p) for p in recovered.rglob("*") if p.is_file()} != expected:
            raise RuntimeError(f"SHA256 roundtrip mismatch: {category}, {mode}")
        report["codecs"][mode]={"bytes":dest.stat().st_size,"seconds":took,
                                "ratio":round(raw/dest.stat().st_size,3)}
        shutil.rmtree(output)
    tarpath = directory/f"{category}.tar"
    with tarfile.open(tarpath,"w") as tf:
        for p in files:
            tf.add(p,arcname=p.relative_to(source).as_posix(),recursive=False)
    if shutil.which("xz"):
        target=directory/f"{category}.tar.xz"
        with target.open("wb") as out:
            took=run(["xz","-9","-T1","-c",str(tarpath)],stdout=out)
        run(["xz","-t",str(target)])
        report["codecs"]["tar.xz -9"]={"bytes":target.stat().st_size,"seconds":took,
                                     "ratio":round(raw/target.stat().st_size,3)}
    if shutil.which("zstd"):
        target=directory/f"{category}.tar.zst"
        took=run(["zstd","--ultra","-19","-q","-f",str(tarpath),"-o",str(target)])
        run(["zstd","-t","-q",str(target)])
        report["codecs"]["tar.zstd -19"]={"bytes":target.stat().st_size,"seconds":took,
                                       "ratio":round(raw/target.stat().st_size,3)}
    tarpath.unlink()
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binary",type=Path,required=True)
    ap.add_argument("--workdir",type=Path,required=True)
    ap.add_argument("--scale",type=int,default=1)
    args=ap.parse_args()
    if args.scale<1 or args.scale>8: ap.error("scale must be 1..8")
    binary=args.binary.resolve()
    workdir=args.workdir.resolve()
    if workdir.exists(): ap.error("workdir already exists: use a fresh directory")
    workdir.mkdir(parents=True)
    report={"seed":SEED,"platform":platform.platform(),"scale":args.scale,
            "methods":"DedupArc maximum/extreme, tar.xz -9, tar.zstd -19",
            "note":"Synthetic repeated versions designed to demonstrate delta coding; not a general compression ranking. tar+codec times exclude tar assembly.",
            "datasets":[]}
    for name in ("versioned","random","text"):
        result=benchmark(binary,workdir,name,args.scale)
        report["datasets"].append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    (workdir/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("RESULTS:",workdir/"results.json")

if __name__=="__main__":
    main()
