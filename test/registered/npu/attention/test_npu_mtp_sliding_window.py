"""Numerical coverage of MTP window bounds in both paged FIA operators.

Run on one NPU with the NPU SGLang dependencies installed:
python test/registered/npu/attention/test_npu_mtp_sliding_window.py -v
"""

import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    AscendAttnMaskBuilder,
    ForwardMetadata,
)
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=60, suite="full-1-npu-a3", nightly=True)


class TestNpuMtpSlidingWindow(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = AscendAttnMaskBuilder.generate_mask_flag(2048).to("npu")

    def _inputs(self, q_lens, kv_lens, heads=4, kv_heads=1, dim=128):
        generator = torch.Generator().manual_seed(46920)
        q = torch.randn(sum(q_lens), heads, dim, generator=generator).bfloat16()
        k = [
            torch.randn(n, kv_heads, dim, generator=generator).bfloat16()
            for n in kv_lens
        ]
        v = [torch.randn(x.shape, generator=generator).bfloat16() for x in k]
        return q, k, v

    def _run(
        self,
        q,
        keys,
        values,
        q_lens,
        window,
        encoder,
        hybrid,
        verify=False,
        graph_metadata=False,
    ):
        # Prepopulate a paged cache, reversing physical pages so the test also
        # exercises block-table indirection rather than a contiguous KV shortcut.
        page_size = 128
        kv_lens = [x.shape[0] for x in keys]
        page_counts = [(n + page_size - 1) // page_size for n in kv_lens]
        total_pages = sum(page_counts)
        kv_heads, dim = keys[0].shape[1:]
        k_cache = torch.zeros(total_pages, page_size, kv_heads, dim, dtype=q.dtype)
        v_cache = torch.zeros_like(k_cache)
        table = torch.zeros(len(q_lens), max(page_counts), dtype=torch.int32)
        page_offset = 0
        for batch, (k, v, count) in enumerate(zip(keys, values, page_counts)):
            for logical in range(count):
                physical = total_pages - 1 - page_offset - logical
                start = logical * page_size
                length = min(page_size, k.shape[0] - start)
                table[batch, logical] = physical
                k_cache[physical, :length] = k[start : start + length]
                v_cache[physical, :length] = v[start : start + length]
            page_offset += count
        k_cache = k_cache.to("npu")
        v_cache = v_cache.to("npu")
        table = table.to("npu")

        # Avoid constructing a model runner; execute the real imported backend
        # method and real NPU operators with only their required state.
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.use_mla = False
        backend.page_size = page_size
        backend.is_hybrid_swa = hybrid
        backend.graph_mode = graph_metadata
        backend.speculative_num_draft_tokens = 4
        backend.mtp_mask = self.mask
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: k_cache,
            get_value_buffer=lambda _: v_cache,
        )
        backend.forward_metadata = ForwardMetadata(
            block_tables=table,
            block_tables_swa=table,
            seq_lens_cpu_int=(
                None if graph_metadata else torch.tensor(kv_lens, dtype=torch.int32)
            ),
            seq_lens_cpu_list=kv_lens,
        )
        batch = SimpleNamespace(
            forward_mode=(
                ForwardMode.TARGET_VERIFY if verify else ForwardMode.DRAFT_EXTEND_V2
            ),
            global_num_token_non_padded_cpu=sum(q_lens),
            extend_seq_lens_cpu=q_lens,
            _original_batch_size=len(q_lens),
        )
        layer = RadixAttention(
            num_heads=q.shape[1],
            num_kv_heads=kv_heads,
            head_dim=dim,
            scaling=dim**-0.5,
            layer_id=0,
            sliding_window_size=window,
            attn_type=AttentionType.ENCODER_ONLY if encoder else AttentionType.DECODER,
        )
        return (
            backend.forward_mtp(
                q.to("npu"), None, None, layer, batch, save_kv_cache=False
            )
            .cpu()
            .float()
            .reshape(q.shape)
        )

    def _reference(self, q, keys, values, q_lens, window, encoder):
        outputs = []
        offset = 0
        for q_len, k, v in zip(q_lens, keys, values):
            query = q[offset : offset + q_len].float().transpose(0, 1)
            repeat = q.shape[1] // k.shape[1]
            key = k.float().repeat_interleave(repeat, dim=1).transpose(0, 1)
            value = v.float().repeat_interleave(repeat, dim=1).transpose(0, 1)
            # FIA aligns the last query with the last KV token for cached prefixes.
            rows = torch.arange(q_len)[:, None] + k.shape[0] - q_len
            cols = torch.arange(k.shape[0])[None, :]
            allowed = torch.ones(q_len, k.shape[0], dtype=torch.bool)
            if window != -1:
                allowed &= (rows - cols).abs() <= window
            if not encoder:
                allowed &= cols <= rows
            scores = query @ key.transpose(-1, -2) * q.shape[-1] ** -0.5
            scores.masked_fill_(~allowed, float("-inf"))
            outputs.append((scores.softmax(-1) @ value).transpose(0, 1))
            offset += q_len
        return torch.cat(outputs)

    def _check(self, q_lens, kv_lens, window, encoder, **kwargs):
        q, k, v = self._inputs(q_lens, kv_lens)
        expected = self._reference(q, k, v, q_lens, window, encoder)
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid, q_lens=q_lens, kv_lens=kv_lens):
                actual = self._run(q, k, v, q_lens, window, encoder, hybrid, **kwargs)
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    def test_encoder_sliding_window(self):
        self._check([128], [128], window=31, encoder=True)

    def test_encoder_global(self):
        self._check([37, 19], [128, 257], window=-1, encoder=True)

    def test_decoder_sliding_window(self):
        self._check([128], [128], window=31, encoder=False)

    def test_decoder_global(self):
        self._check([37, 19], [128, 257], window=-1, encoder=False)

    def test_right_aligned_prefix(self):
        self._check([16], [256], window=31, encoder=True)

    def test_ragged_batch(self):
        self._check([37, 19], [128, 257], window=31, encoder=True)

    def test_window_boundaries(self):
        for length in (31, 32, 33):
            with self.subTest(length=length):
                self._check([length], [length], window=31, encoder=True)

    def test_target_verify(self):
        self._check([4, 4], [128, 257], window=31, encoder=True, verify=True)

    def test_graph_metadata(self):
        # Test the list-based metadata branch, not graph capture/replay itself.
        self._check(
            [4, 4],
            [128, 257],
            window=31,
            encoder=True,
            verify=True,
            graph_metadata=True,
        )

    def test_future_visibility_and_outside_window_isolation(self):
        q = torch.zeros(128, 4, 128, dtype=torch.bfloat16)
        k = [torch.zeros(128, 1, 128, dtype=torch.bfloat16)]
        v = [torch.zeros_like(k[0])]
        # At query 64, keys 33 and 95 are exactly on the symmetric bounds.
        # Key 96 must have no effect. Causal or global attention both fail.
        v[0][33] = 1
        v[0][95] = 2
        v[0][96] = 64
        expected = torch.full((4, 128), 3 / 63)
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                actual = self._run(q, k, v, [128], 31, True, hybrid)
                torch.testing.assert_close(actual[64], expected, atol=2e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
