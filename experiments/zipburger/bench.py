"""Run deterministic standalone ZipBurger R01 evaluation: python -m experiments.zipburger.bench."""
import json
import time
import zlib

import numpy as np

from .neurocrunch import pack


def corpora():
    rng = np.random.default_rng(14014)
    roots = rng.integers(-12, 13, (8, 256), dtype=np.int8)
    base_rows = [v.copy() for v in roots]
    sparse_rows = base_rows.copy()
    dense_rows = base_rows.copy()
    for i in range(256):
        b = roots[i % 8].astype(np.int16)
        row = (b * (1, 2, 4, 8)[(i // 8) % 4]).astype(np.int8)
        perturbed = row.copy()
        for p in range(1 + i % 4):
            j = (i * 97 + p * 31) % 256
            perturbed[j] = np.int8(int(perturbed[j]) + 1)
        sparse_rows.append(perturbed)
        dense = row.copy()
        for p in range(64):
            j = (i * 97 + p * 31) % 256
            dense[j] = np.int8(int(dense[j]) + 1)
        dense_rows.append(dense)
    yield "shared-roots+sparse-edits", np.stack(sparse_rows)
    yield "shared-roots+dense-edits", np.stack(dense_rows)
    yield "random-int8", rng.integers(-128, 128, (264, 256), dtype=np.int16).astype(np.int8)
    yield "zeros", np.zeros((264, 256), dtype=np.int8)


def main():
    rng = np.random.default_rng(100)
    reports = []
    for label, w in corpora():
        t0 = time.perf_counter()
        carrier = pack(w)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        x = rng.integers(-10, 11, w.shape[1], dtype=np.int64)
        assert np.array_equal(carrier.dot(x), w.astype(np.int64) @ x)
        assert carrier.restore().tobytes() == w.tobytes()
        reports.append({"corpus": label, "raw_bytes": w.nbytes,
                        "zipburger_bytes": carrier.compressed_bytes,
                        "zlib9_bytes": len(zlib.compress(w.tobytes(), level=9)),
                        "roots": carrier.root_count, "linked_rows": carrier.link_count,
                        "residual_edges": carrier.residual_count,
                        "pack_ms": ms, "exact_roundtrip": True, "exact_integer_gemv": True})
    try:
        import torch
        from .neurocrunch import make_torch_module
        w = next(corpora())[1]
        c = pack(w)
        net = make_torch_module(c)
        parameter_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
        tensor_bytes = (net.root_neurons.numel() * net.root_neurons.element_size()
                        + net.row_root_ids.numel() * net.row_root_ids.element_size()
                        + net.row_gain.numel() * net.row_gain.element_size()
                        + net.correction_edges._indices().numel() * net.correction_edges._indices().element_size()
                        + net.correction_edges._values().numel() * net.correction_edges._values().element_size())
        reports[0]["trainable_adapter_bytes_fp32"] = parameter_bytes
        reports[0]["torch_core_buffer_bytes_estimate"] = tensor_bytes
        reports[0]["serialized_frame_plus_fp32_adapters"] = c.compressed_bytes + parameter_bytes
    except ImportError:
        pass
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
