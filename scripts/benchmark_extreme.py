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
    elif category == "math":
        length = 2 * 1024 * 1024 * scale
        pattern = bytes(range(256))
        (root / "affine.bin").write_bytes(pattern * (length // 256))
        (root / "constant.bin").write_bytes(bytes([0x7d]) * length)
        (root / "periodic.bin").write_bytes((b"abcdefg" * ((length + 6) // 7))[:length])
        patched = bytearray(pattern * (length // 256))
        for _ in range(100 * scale):
            at = rng.randrange(len(patched))
            patched[at] ^= rng.randint(1, 255)
        (root / "affine-patched.bin").write_bytes(patched)
    elif category == "matrix":
        # Matrix data with a non-periodic 128-byte column basis and
        # per-row quadratic coefficients modulo 256. The column basis
        # is deliberately random: traditional R02 formula patterns
        # cannot simply repeat a 32-byte periodic template.
        width, rows = 128, 4096 * scale
        for variant in range(4):
            basis = rng.randbytes(width)
            data = bytearray(rows*width)
            for row in range(rows):
                offset = (variant * 17 + row*11 + 5*(row*(row-1)//2)) & 255
                start = row*width
                for col in range(width):
                    data[start+col] = (basis[col] + offset) & 255
            for _ in range(30 + variant*10):
                pos = width + rng.randrange(len(data)-width)
                data[pos] ^= rng.randint(1,255)
            (root / f"matrix-{variant:02}.bin").write_bytes(data)
    elif category == "source-tree":
        # Real, versioned source-code snapshot from this Git checkout.
        # Avoid ephemeral build outputs and vendored/binary files.
        checkout = Path(__file__).resolve().parents[1]
        allowed = {".go", ".md", ".txt", ".py", ".yml", ".yaml", ".mod", ".ps1", ".json"}
        selected = []
        for section in ("internal", "cmd", "scripts", "docs", ".github"):
            for p in (checkout / section).rglob("*"):
                if p.is_file() and p.suffix in allowed and p.stat().st_size <= 4*1024*1024:
                    selected.append(p)
        selected.sort()
        for p in selected:
            dst = root / p.relative_to(checkout)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        shutil.copyfile(checkout / "README.md", root / "README.md")
    elif category == "quadratic":
        # Integer-only quadratic polynomial modulo 256. The period is
        # longer than the R02 32-byte period cap.
        length = 2 * 1024 * 1024 * scale
        for variant, (a, b, c) in enumerate(((13, 17, 5), (91, 7, 11))):
            data = bytearray(length)
            for i in range(length):
                data[i] = (a + b*i + c*(i*(i-1)//2)) & 255
            if variant:
                for _ in range(50*scale):
                    at = rng.randrange(length)
                    data[at] ^= rng.randint(1,255)
            (root / f"quadratic-{variant}.bin").write_bytes(data)
    elif category == "shared-math":
        # Distinct files share an exactly regenerable periodic root with sparse edits.
        # Kept separate from the R02 'math' set to measure cross-file references.
        length = 512 * 1024 * scale
        pattern = b"deduparc-neutron-dag-period-31!"
        root_data = bytes(pattern[i % len(pattern)] for i in range(length))
        for n in range(8):
            data = bytearray(root_data)
            for j in range(2*n):
                pos = 1024 + j * 17011
                if pos < length:
                    data[pos] ^= (11+n+j)&255
            (root / f"formula-{n:02}.bin").write_bytes(data)
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
    for mode in ("maximum","extreme","neutron","neutron-dag","neutron-dense","neutron-rmg"):
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
            "methods":"DedupArc maximum/extreme/neutron/neutron-dag/neutron-dense/neutron-rmg, tar.xz -9, tar.zstd -19",
            "note":"Synthetic corpora: versioned for delta coding, math for formula coding, shared-math for cross-file formulas, quadratic for v6-only integer polynomials, source-tree for this checkout's real files, text/random for negative controls. Not a general compression ranking. tar+codec times exclude tar assembly.",
            "datasets":[]}
    for name in ("versioned","random","text","math","shared-math","quadratic","matrix","source-tree"):
        result=benchmark(binary,workdir,name,args.scale)
        report["datasets"].append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    (workdir/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("RESULTS:",workdir/"results.json")

if __name__=="__main__":
    main()
