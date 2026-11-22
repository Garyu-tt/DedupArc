"""Bounded HTTPS range benchmark of actual GPT-OSS-20B GGUF MXFP4 blocks.

Reads ONLY bounded ranges (two tiny samples of up to nine tensors) from a pinned public GGUF.
Never downloads a checkpoint, touches Mikoshi, or modifies inference services.
This uses the GGUF v3 tensor offsets independently observed on VM200. Fail
closed on non-206 responses, oversized reads, wrong Content-Range, redirects
outside trusted HF hosts, or unexpected tensor bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import re
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib

import numpy as np

from .mxfp4_r03 import dense_reference_dot, load, pack
from .mxfp4_palette_r04 import pack as pack_scale_bitpacked_r04

REVISION = "41e8a94d15d62070b36cf4553a0250717ecf46b8"
FILE = "gpt-oss-20b-MXFP4.gguf"
URL = f"https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/{REVISION}/{FILE}"
EXPECTED_FILE_BYTES = 12_109_566_624
EXPECTED_WHOLE_FILE_SHA256 = (
    "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901"
)
GGUF_DATA_START = 13_008_288
GROUPS_PER_ROW = 2880 // 32
BLOCK_BYTES = 17
# Measured from the pinned GGUF v3 header on VM200. Offsets are relative
# to GGUF_DATA_START, NOT absolute file offsets.
TENSORS = {
    "blk.0.ffn_down_exps.weight": 1_259_283_456,
    "blk.0.ffn_gate_exps.weight": 1_400_656_896,
    "blk.0.ffn_up_exps.weight": 1_542_399_104,
    "blk.1.ffn_down_exps.weight": 1_712_028_800,
    "blk.1.ffn_gate_exps.weight": 1_853_402_240,
    "blk.1.ffn_up_exps.weight": 1_995_144_448,
    "blk.2.ffn_down_exps.weight": 2_164_774_144,
    "blk.2.ffn_gate_exps.weight": 2_306_147_584,
    "blk.2.ffn_up_exps.weight": 2_447_889_792,
}
MAX_RANGE_BYTES = 120_000


class RangeProbeError(ValueError):
    pass


def _trusted_host(url):
    p = urlsplit(url)
    h = (p.hostname or "").lower()
    return p.scheme == "https" and (
        h == "huggingface.co" or h.endswith(".huggingface.co")
        or h.endswith(".hf.co")) and p.username is None and p.password is None


class _TrustedHFRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _trusted_host(newurl):
            raise RangeProbeError("untrusted HTTP redirect destination")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def get_range(first: int, last: int, *, opener=None) -> bytes:
    if (type(first) is not int or type(last) is not int or first < 0
            or last < first or last >= EXPECTED_FILE_BYTES
            or last - first + 1 > MAX_RANGE_BYTES):
        raise RangeProbeError("invalid/oversized GGUF byte range")
    request = Request(URL, headers={
        "Range": f"bytes={first}-{last}", "Accept-Encoding": "identity",
        "User-Agent": "NeuroCrunchZipBurger-RealRangeResearch/1.0",
    })
    client = opener if opener is not None else build_opener(_TrustedHFRedirect())
    with client.open(request, timeout=30) as response:
        if not _trusted_host(response.geturl()):
            raise RangeProbeError("download left trusted Hugging Face hosts")
        if response.status != 206:
            raise RangeProbeError(f"HTTP {response.status}, required HTTP 206")
        expected_range = f"bytes {first}-{last}/{EXPECTED_FILE_BYTES}"
        content_range = response.headers.get("Content-Range", "")
        if content_range != expected_range:
            raise RangeProbeError(
                f"unexpected Content-Range: {content_range!r}; expected {expected_range!r}")
        length = last - first + 1
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            if not content_length.isdecimal() or int(content_length) != length:
                raise RangeProbeError("unexpected content length")
        payload = response.read(length + 1)
        if len(payload) != length:
            raise RangeProbeError("truncated or oversized range response")
    return payload


def sample(name: str, row_start=0, rows=48, groups=4, *, fetch=get_range):
    if name not in TENSORS:
        raise RangeProbeError("tensor is not in the pinned GGUF manifest")
    if (type(row_start) is not int or type(rows) is not int
            or type(groups) is not int or row_start < 0
            or row_start + rows > 2880 * 32 or not 1 <= rows <= 128
            or not 1 <= groups <= 16):
        raise RangeProbeError("sample is outside bounded MXFP4 tensor")
    row_stride = GROUPS_PER_ROW * BLOCK_BYTES
    absolute_start = GGUF_DATA_START + TENSORS[name] + row_start * row_stride
    absolute_end = absolute_start + (rows - 1) * row_stride + groups * BLOCK_BYTES - 1
    if absolute_end - absolute_start + 1 > MAX_RANGE_BYTES:
        raise RangeProbeError("sample exceeds strict network budget")
    span = fetch(absolute_start, absolute_end)
    if len(span) != absolute_end - absolute_start + 1:
        raise RangeProbeError("sample transport returned wrong size")
    raw = bytearray()
    for i in range(rows):
        offset = i * row_stride
        raw.extend(span[offset:offset + groups * BLOCK_BYTES])
    data = np.frombuffer(raw, dtype=np.uint8).reshape(rows, groups, BLOCK_BYTES)
    scales = np.ascontiguousarray(data[:, :, 0])
    blocks = np.ascontiguousarray(data[:, :, 1:])
    return blocks, scales, bytes(raw), (absolute_start, absolute_end)


def benchmark(name, row_start=0, *, rows=48, groups=4, fetch=get_range):
    blocks, scales, packed_gguf, (first, last) = sample(
        name, row_start, rows=rows, groups=groups, fetch=fetch)
    start = time.perf_counter()
    carrier = pack(blocks, scales, max_roots=8, max_candidates=16)
    pack_ms = (time.perf_counter() - start) * 1000
    stored = load(carrier.payload)
    rb, rs = stored.restore()
    reconstructed = np.empty((rows, groups, 17), dtype=np.uint8)
    reconstructed[:, :, 0] = rs
    reconstructed[:, :, 1:] = rb
    if reconstructed.tobytes() != packed_gguf:
        raise AssertionError("R03 failed original GGUF bitwise restoration")
    x = np.random.default_rng(331).standard_normal(groups * 32)
    start = time.perf_counter()
    direct = stored.dot(x)
    dot_ms = (time.perf_counter() - start) * 1000
    oracle = dense_reference_dot(blocks, scales, x)
    np.testing.assert_allclose(direct, oracle, rtol=1e-12, atol=1e-8)

    # R04 never expands an entire floating-point tensor: only one 32-value
    # MXFP4 block is decoded per direct GEMV step. Compare exact same bytes.
    start = time.perf_counter()
    palette = pack_scale_bitpacked_r04(blocks, scales)
    palette_pack_ms = (time.perf_counter() - start) * 1000
    bp, sp = palette.restore()
    if not np.array_equal(bp, blocks) or not np.array_equal(sp, scales):
        raise AssertionError("R04 failed exact GGUF scale/code restoration")
    start = time.perf_counter()
    palette_direct = palette.dot(x)
    palette_direct_ms = (time.perf_counter() - start) * 1000
    np.testing.assert_allclose(palette_direct, oracle, rtol=1e-12, atol=1e-8)
    return {
        "tensor": name, "row_start": row_start,
        "source_gguf_range": [first, last],
        "sample_sha256": hashlib.sha256(packed_gguf).hexdigest(),
        "raw_bytes": len(packed_gguf),
        "zipburger_r03_bytes": carrier.frame_bytes,
        "zipburger_mode": carrier.mode,
        "zipburger_r04_bytes": palette.frame_bytes,
        "zipburger_r04_mode": palette.mode,
        "zipburger_r04_over_raw": round(palette.frame_bytes / len(packed_gguf), 4),
        "zipburger_r04_pack_ms": round(palette_pack_ms, 3),
        "zipburger_r04_direct_dot_ms": round(palette_direct_ms, 3),
        "zipburger_r04_exact_packed_bits": True,
        "zipburger_r04_float64_direct_dot_reference_match": True,
        "scale_unique_values": int(np.unique(scales).size),
        "scale_bytes": int(scales.nbytes),
        "scale_zlib9_bytes": len(zlib.compress(scales.tobytes(), 9)),
        "mantissa_bytes": int(blocks.nbytes),
        "mantissa_zlib9_bytes": len(zlib.compress(blocks.tobytes(), 9)),
        "zlib9_bytes": len(zlib.compress(packed_gguf, 9)),
        "xz6_bytes": len(lzma.compress(packed_gguf, preset=6)),
        "zipburger_over_raw": round(carrier.frame_bytes / len(packed_gguf), 4),
        "pack_ms": round(pack_ms, 3), "direct_dot_ms": round(dot_ms, 3),
        "bitwise_gguf_restoration": True,
        "float64_direct_dot_reference_match": True,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-start", type=int, action="append",
                    help="repeat for multiple row positions (default: 0 and 4096)")
    ap.add_argument("--tensor", action="append",
                    help="repeat for a few names in the pinned manifest")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--groups", type=int, default=4)
    args = ap.parse_args(argv)
    starts = args.row_start if args.row_start is not None else [0, 4096]
    names = args.tensor if args.tensor is not None else list(TENSORS)
    if not starts or not names or len(starts) * len(names) > 24:
        ap.error("requires 1..24 total bounded samples")
    results = [benchmark(name, row, rows=args.rows, groups=args.groups)
               for name in names for row in starts]
    report = {
        "schema": "zipburger-mxfp4-real-public-gguf-ranges/1",
        "source_repo": "ggml-org/gpt-oss-20b-GGUF",
        "pinned_revision": REVISION, "gguf_filename": FILE,
        "expected_whole_file_sha256_from_upstream_metadata": EXPECTED_WHOLE_FILE_SHA256,
        "whole_file_hash_recomputed": False,
        "gguf_bytes": EXPECTED_FILE_BYTES,
        "real_public_checkpoint_bytes": True,
        "model_inference_performed": False,
        "gpu_involved": False,
        "whole_model_compression_tested": False,
        "results": results,
    }
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
