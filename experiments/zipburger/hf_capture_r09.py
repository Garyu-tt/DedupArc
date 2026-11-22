"""R09 opt-in local Hugging Face GPT-OSS-20B activation capture.

This is a separate research command, never invoked in CI. It requires an
ALREADY-LOCAL transformers checkpoint, cannot consume GGUF directly and does
not contact any inference service. No prompt strings or model weights are
saved; numerical activations can still be sensitive, so keep them private.

Current HF GptOssModel exposes model.layers, layer.mlp.router and
layer.mlp.experts. Fused/kernalized backends might bypass expected hooks:
fail closed if any hook does not fire. Expert up/gate *inputs* are captured;
down-projection inputs are NOT available through these hooks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .layer_trace_r09 import LAYERS, Trace, save

TILE_INPUT_WIDTH = 512
MAX_PROMPTS = 8
MAX_TOKENS_PER_PROMPT = 32
MAX_EXPERT_OBSERVATIONS = 256


def _digest_prompts(prompts):
    data = json.dumps(prompts, ensure_ascii=False, sort_keys=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def select_expert_inputs(inputs, route, expert_id, width=TILE_INPUT_WIDTH):
    """Return only actual routed input vectors; never synthesize missing routes."""
    x = np.asarray(inputs)
    r = np.asarray(route)
    if (x.ndim != 2 or r.ndim != 2 or r.shape != (x.shape[0], 4)
            or not 1 <= width <= x.shape[1]
            or not isinstance(expert_id, int) or not 0 <= expert_id < 32):
        raise ValueError("invalid expert activation selector")
    if not np.all(np.isfinite(x)) or np.any(r < 0) or np.any(r >= 32):
        raise ValueError("bad routed expert inputs")
    chosen = np.any(r == expert_id, axis=1)
    return np.ascontiguousarray(x[chosen, :width], dtype=np.float32)


def _to_cpu_array(value, *, dtype):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def collect(model, tokenizer, prompts, *, checkpoint_sha256, source,
            max_tokens=MAX_TOKENS_PER_PROMPT, device="cpu"):
    """Capture one bounded teacher (or candidate) prefill with 24 layer hooks.

    Full logits of a short prompt are evaluated to get argmax; no generation.
    This is a _model_ capture, NOT an agency/tool-use benchmark.
    """
    import torch  # explicit optional dependency for real model runs only

    if (len(prompts) < 1 or len(prompts) > MAX_PROMPTS
            or any(not isinstance(p, str) or not p or len(p) > 8192 for p in prompts)
            or not 1 <= max_tokens <= MAX_TOKENS_PER_PROMPT):
        raise ValueError("invalid bounded prompt corpus")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("operator must supply actual checkpoint SHA-256")
    layers = getattr(getattr(model, "model", None), "layers", None)
    cfg = getattr(model, "config", None)
    if (layers is None or len(layers) != LAYERS or cfg is None
            or getattr(cfg, "num_hidden_layers", None) != LAYERS
            or getattr(cfg, "num_local_experts", None) != 32
            or getattr(cfg, "num_experts_per_tok", None) != 4):
        raise ValueError("not a supported GPT-OSS-20B 24-layer MoE model")
    if source not in ("instrumented_teacher", "instrumented_candidate"):
        raise ValueError("invalid capture role")

    buffer_hidden = [[] for _ in range(LAYERS)]
    buffer_routes = [[] for _ in range(LAYERS)]
    buffer_expert = {(layer, expert): [] for layer in range(LAYERS)
                     for expert in range(32)}
    current_hidden, current_route, current_expert = {}, {}, {}
    handles = []

    def hidden_hook(index):
        def callback(module, args, output):
            y = output[0] if isinstance(output, (tuple, list)) else output
            values = _to_cpu_array(y, dtype=np.float32)
            if values.ndim != 3 or values.shape[0] != 1:
                raise ValueError("unexpected hidden state capture shape")
            current_hidden[index] = values.reshape(-1, values.shape[-1]).copy()
        return callback

    def router_hook(index):
        def callback(module, args, output):
            if not isinstance(output, (tuple, list)) or len(output) < 3:
                raise ValueError("router hook did not expose top-4 indices")
            values = _to_cpu_array(output[2], dtype=np.int16)
            if values.ndim != 2 or values.shape[1] != 4:
                raise ValueError("unexpected router top4 capture shape")
            current_route[index] = values.copy()
        return callback

    def experts_hook(index):
        def callback(module, args):
            if len(args) < 3:
                raise ValueError("expert inputs/router not exposed")
            values = _to_cpu_array(args[0], dtype=np.float32)
            chosen = _to_cpu_array(args[1], dtype=np.int16)
            if values.ndim != 2 or chosen.shape != (values.shape[0], 4):
                raise ValueError("unexpected routed expert pre-input shape")
            if values.shape[1] < TILE_INPUT_WIDTH:
                raise ValueError("not enough expert input coordinates")
            for expert in np.unique(chosen):
                idx = int(expert)
                if not 0 <= idx < 32:
                    raise ValueError("unexpected expert ID")
                current_expert[(index, idx)] = select_expert_inputs(values, chosen, idx)
        return callback

    try:
        for i, layer in enumerate(layers):
            handles.extend([
                layer.register_forward_hook(hidden_hook(i)),
                layer.mlp.router.register_forward_hook(router_hook(i)),
                layer.mlp.experts.register_forward_pre_hook(experts_hook(i)),
            ])
        all_ids, all_preds = [], []
        next_id = 0
        model.eval()
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=max_tokens)
            token_ids = encoded["input_ids"]
            if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] < 1:
                raise ValueError("unexpected tokenizer result")
            # No implicit device-map or GPU use. The caller loads on one
            # chosen device and explicitly controls offload policy.
            encoded = {k: v.to(device) for k, v in encoded.items()}
            current_hidden.clear()
            current_route.clear()
            current_expert.clear()
            with torch.inference_mode():
                result = model(**encoded, use_cache=False)
            pred = result.logits.argmax(dim=-1).reshape(-1)
            pred = pred.detach().cpu().numpy().astype(np.int64)
            n = pred.size
            if (len(current_hidden) != LAYERS or len(current_route) != LAYERS
                    or len({layer for layer, _ in current_expert}) != LAYERS
                    or n > max_tokens):
                raise RuntimeError("24-layer capture incomplete; backend hooks may be bypassed")
            for i in range(LAYERS):
                h, r = current_hidden[i], current_route[i]
                if h.shape[0] != n or r.shape != (n, 4):
                    raise RuntimeError("teacher hidden/router token alignment failed")
                buffer_hidden[i].append(h)
                buffer_routes[i].append(r)
            for key, values in current_expert.items():
                if values.size and sum(v.shape[0] for v in buffer_expert[key]) < MAX_EXPERT_OBSERVATIONS:
                    buffer_expert[key].append(values)
            all_preds.append(pred)
            all_ids.append(np.arange(next_id, next_id + n, dtype=np.int64))
            next_id += n
        trace = Trace(
            np.concatenate(all_ids),
            tuple(np.concatenate(rows).astype(np.float32) for rows in buffer_hidden),
            tuple(np.concatenate(rows).astype(np.int16) for rows in buffer_routes),
            np.concatenate(all_preds),
            _digest_prompts(prompts), checkpoint_sha256, source,
        )
        experts = {f"layer_{layer:02d}_expert_{expert:02d}": np.concatenate(values)[:MAX_EXPERT_OBSERVATIONS]
                   for (layer, expert), values in buffer_expert.items() if values}
        return trace, experts
    finally:
        for handle in handles:
            handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model-dir", type=Path, required=True,
                        help="existing local Transformers checkpoint, not GGUF")
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--trace-npz", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source", required=True,
                        choices=("instrumented_teacher", "instrumented_candidate"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    prompts = json.loads(args.prompts_json.read_text(encoding="utf-8"))
    if not isinstance(prompts, list):
        raise ValueError("prompt file must be a JSON array of strings")
    if not args.local_model_dir.is_dir():
        raise ValueError("checkpoint directory does not exist")
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError("invalid prompt count")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Deliberately no network downloads, no trust_remote_code, no writebacks.
    tokenizer = AutoTokenizer.from_pretrained(str(args.local_model_dir),
                                              local_files_only=True,
                                              trust_remote_code=False)
    # Reject same-tokenized prompts even if prompt text ordering differs.
    prompt_token_hashes = []
    for text in prompts:
        if not isinstance(text, str) or not text or len(text) > 8192:
            raise ValueError("invalid prompt text")
        token_tensor = tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=MAX_TOKENS_PER_PROMPT)["input_ids"]
        token_bytes = token_tensor.detach().cpu().numpy().astype("<i8").tobytes()
        prompt_token_hashes.append(hashlib.sha256(token_bytes).hexdigest())
    if len(set(prompt_token_hashes)) != len(prompt_token_hashes):
        raise ValueError("repeated tokenized prompt in corpus")
    model = AutoModelForCausalLM.from_pretrained(str(args.local_model_dir),
                                                local_files_only=True,
                                                trust_remote_code=False)
    model.to(args.device)
    trace, expert_inputs = collect(
        model, tokenizer, prompts, checkpoint_sha256=args.checkpoint_sha256,
        source=args.source, device=args.device)
    manifest = save(trace, args.trace_npz)
    if expert_inputs:
        expert_file = args.trace_npz.with_name(args.trace_npz.stem + "_expert_upgate.npz")
        np.savez_compressed(expert_file, **expert_inputs)
        expert_meta = {
            "schema": "zipburger-r09-expert-inputs-v1",
            "sha256": hashlib.sha256(expert_file.read_bytes()).hexdigest(),
            "prompt_digest": manifest["prompt_digest"],
            "token_sequence_sha256": prompt_token_hashes,
            "checkpoint_digest": manifest["checkpoint_digest"],
            "source": args.source,
            "input_width": TILE_INPUT_WIDTH,
            "projections": ["up", "gate"],
            "contains_down_projection_inputs": False,
            "real_teacher_activations_claim": args.source == "instrumented_teacher",
            "verified_full_model_agenticity": False,
        }
        expert_file.with_suffix(".json").write_text(json.dumps(expert_meta, sort_keys=True, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps({**manifest, "expert_upgate_groups": len(expert_inputs),
                      "down_projection_inputs_recorded": False}, indent=2))


if __name__ == "__main__":
    main()
