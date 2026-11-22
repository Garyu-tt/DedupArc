"""Offline protocol and exact-MXFP4 tests for the public GGUF range probe."""
import io
import re
import unittest
from unittest.mock import Mock

import numpy as np

from experiments.zipburger import real_mxfp4_ranges_r04 as r


class FakeResponse:
    status = 206

    def __init__(self, start, end, payload=None, *, code=206, content_range=None,
                 content_length=None, url=r.URL):
        self.status = code
        self.url = url
        self.content = payload if payload is not None else b"z" * (end - start + 1)
        self.headers = {
            "Content-Range": (
                f"bytes {start}-{end}/{r.EXPECTED_FILE_BYTES}"
                if content_range is None else content_range),
            "Content-Length": (
                str(len(self.content)) if content_length is None
                else content_length),
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def geturl(self):
        return self.url

    def read(self, size):
        return self.content[:size]


class FakeOpener:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_request = None

    def open(self, request, timeout):
        self.calls += 1
        self.last_request = request
        return self.response


class RangeProbeTests(unittest.TestCase):
    def test_rejects_unsafe_ranges_before_open(self):
        for begin, end in (
            (-1, 100), (12, 11), (0, r.MAX_RANGE_BYTES),
            (0, r.EXPECTED_FILE_BYTES), (True, 8),
            (0, False),
        ):
            with self.subTest(begin=begin, end=end):
                client = Mock()
                with self.assertRaises(r.RangeProbeError):
                    r.get_range(begin, end, opener=client)
                client.open.assert_not_called()

    def test_exact_range_and_fixed_revision(self):
        start, end = 90, 105
        client = FakeOpener(FakeResponse(start, end))
        data = r.get_range(start, end, opener=client)
        self.assertEqual(len(data), 16)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.last_request.get_header("Range"),
                         "bytes=90-105")
        self.assertIn(r.REVISION, client.last_request.full_url)

    def test_refuse_200_or_wrong_content_range(self):
        a, b = 3, 10
        for response in (
            FakeResponse(a, b, code=200),
            FakeResponse(a, b, content_range="bytes 2-10/12109566624"),
            FakeResponse(a, b, content_length="1000"),
            FakeResponse(a, b, payload=b"short"),
            FakeResponse(a, b, payload=b"too-many!"),
            FakeResponse(a, b, url="https://example.invalid/leak"),
        ):
            with self.subTest(response=response.__dict__):
                with self.assertRaises(r.RangeProbeError):
                    r.get_range(a, b, opener=FakeOpener(response))

    def test_redirects_outside_hf_are_refused(self):
        handler = r._TrustedHFRedirect()
        with self.assertRaises(r.RangeProbeError):
            handler.redirect_request(None, None, 302, "no", {},
                                     "https://localhost/intranet")
        with self.assertRaises(r.RangeProbeError):
            handler.redirect_request(None, None, 302, "no", {},
                                     "http://cas-bridge.xethub.hf.co/model")
        self.assertTrue(r._trusted_host(
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc"))
        self.assertFalse(r._trusted_host(
            "https://cas-bridge.xethub.hf.co.attacker.invalid/"))

    def test_real_shape_interleaved_extract_and_direct_oracle(self):
        rng = np.random.default_rng(4)
        name = "blk.0.ffn_down_exps.weight"
        # Simulate 48 separate GGUF rows of 90 blocks each; the HTTP range
        # includes the intervening groups, not merely the selected 4.
        source = rng.integers(0, 256, (48, r.GROUPS_PER_ROW, 17),
                             dtype=np.uint8)
        source[:, :, 0] = rng.integers(115, 139,
                                       (48, r.GROUPS_PER_ROW), dtype=np.uint8)
        span = source[:47].tobytes() + source[47, :4].tobytes()
        seen = []

        def fetch(first, end):
            seen.append((first, end))
            self.assertEqual(end - first + 1, len(span))
            return span

        blocks, scales, raw, area = r.sample(name, 0, 48, 4, fetch=fetch)
        np.testing.assert_array_equal(blocks, source[:, :4, 1:])
        np.testing.assert_array_equal(scales, source[:, :4, 0])
        self.assertEqual(len(raw), 48 * 4 * 17)
        self.assertEqual(area, seen[0])
        result = r.benchmark(name, 0, fetch=fetch)
        self.assertTrue(result["bitwise_gguf_restoration"])
        self.assertTrue(result["float64_direct_dot_reference_match"])
        self.assertEqual(result["raw_bytes"], 3264)
        self.assertLessEqual(result["zipburger_r03_bytes"], 3308)

    def test_manifest_and_shape_limits(self):
        for kwargs in ({"row_start": -1}, {"row_start": 2880 * 32},
                       {"rows": 0}, {"rows": 129}, {"groups": 0},
                       {"groups": 17}, {"groups": True}, {"rows": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(r.RangeProbeError):
                r.sample("blk.0.ffn_up_exps.weight", **kwargs,
                         fetch=Mock())
        with self.assertRaises(r.RangeProbeError):
            r.sample("blk.23.ffn_up_exps.weight", fetch=Mock())


if __name__ == "__main__":
    unittest.main()
