"""R09 instrumenter hook lifecycle with a fake 24-layer model, no Torch dependency.

Mocks exercise capture integration but are NOT a real GPT-OSS-20B run.
"""
from contextlib import nullcontext
import hashlib
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import numpy as np

from experiments.zipburger import hf_capture_r09 as h


class FakeTensor:
    def __init__(self, array):
        self.data = np.asarray(array)
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def shape(self):
        return self.data.shape
    def detach(self):
        return self
    def float(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.data
    def to(self, device):
        return self
    def argmax(self, dim):
        return FakeTensor(np.argmax(self.data, axis=dim))
    def reshape(self, *shape):
        return FakeTensor(self.data.reshape(*shape))


class HookHandle:
    def __init__(self, owner, hook):
        self.owner, self.hook = owner, hook
    def remove(self):
        self.owner.callbacks.remove(self.hook)


class FakeModule:
    def __init__(self):
        self.callbacks = []
    def register_forward_hook(self, fn):
        self.callbacks.append(fn)
        return HookHandle(self, fn)
    def register_forward_pre_hook(self, fn):
        self.callbacks.append(fn)
        return HookHandle(self, fn)


class FakeLayer(FakeModule):
    def __init__(self):
        super().__init__()
        self.mlp = SimpleNamespace(router=FakeModule(), experts=FakeModule())


class FakeModel:
    def __init__(self, broken_expert_hook=False):
        self.model = SimpleNamespace(layers=[FakeLayer() for _ in range(24)])
        self.config = SimpleNamespace(num_hidden_layers=24,
                                      num_local_experts=32, num_experts_per_tok=4)
        self.broken_expert_hook = broken_expert_hook
    def eval(self):
        pass
    def __call__(self, **encoded):
        n = encoded["input_ids"].shape[1]
        route = np.broadcast_to(np.array([[0, 1, 2, 3]], dtype=np.int16),
                                (n, 4)).copy()
        for idx, layer in enumerate(self.model.layers):
            x = FakeTensor(np.full((1, n, 512), idx / 24, dtype=np.float32))
            for callback in layer.callbacks:
                callback(layer, (x,), x)
            for callback in layer.mlp.router.callbacks:
                callback(layer.mlp.router, (x,),
                         (None, None, FakeTensor(route)))
            if not self.broken_expert_hook:
                for callback in layer.mlp.experts.callbacks:
                    callback(layer.mlp.experts,
                             (FakeTensor(x.data.reshape(n, 512)), FakeTensor(route),
                              FakeTensor(np.full((n, 4), .25, dtype=np.float32))))
        logits = np.zeros((1, n, 8), dtype=np.float32)
        logits[:, :, 2] = 1.0
        return SimpleNamespace(logits=FakeTensor(logits))


def fake_tokenizer(prompt, **kwargs):
    return {"input_ids": FakeTensor(np.array([[1, 2, 3]], dtype=np.int64))}


class InstrumenterHooksR09Tests(unittest.TestCase):
    def test_collects_all_layers_routes_and_real_expert_only(self):
        model = FakeModel()
        checkpoint = hashlib.sha256(b"dummy checkpoint").hexdigest()
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            trace, expert_inputs = h.collect(
                model, fake_tokenizer, ["prompt a", "prompt b"],
                checkpoint_sha256=checkpoint, source="instrumented_teacher")
        self.assertEqual(trace.sample_ids.tolist(), list(range(6)))
        self.assertEqual(len(trace.layers), 24)
        self.assertEqual(trace.layers[0].shape, (6, 512))
        self.assertEqual(trace.predictions.tolist(), [2] * 6)
        self.assertEqual(trace.routes[11].shape, (6, 4))
        self.assertIn("layer_00_expert_00", expert_inputs)
        self.assertIn("layer_23_expert_03", expert_inputs)
        self.assertNotIn("layer_00_expert_04", expert_inputs)
        self.assertEqual(expert_inputs["layer_00_expert_00"].shape, (6, 512))
        for layer in model.model.layers:
            self.assertFalse(layer.callbacks)
            self.assertFalse(layer.mlp.router.callbacks)
            self.assertFalse(layer.mlp.experts.callbacks)

    def test_missing_fused_expert_hook_fails_closed_and_removes_all(self):
        model = FakeModel(broken_expert_hook=True)
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            with self.assertRaises(RuntimeError):
                h.collect(model, fake_tokenizer, ["prompt a"],
                          checkpoint_sha256=hashlib.sha256(b"fake").hexdigest(),
                          source="instrumented_teacher")
        for layer in model.model.layers:
            self.assertFalse(layer.callbacks)
            self.assertFalse(layer.mlp.router.callbacks)
            self.assertFalse(layer.mlp.experts.callbacks)

if __name__ == "__main__":
    unittest.main()
