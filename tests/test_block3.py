"""Fast acceptance tests for the complete Block 3 learning path."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / "exercises" / "block_03_transformer"
PATHS = (
    BLOCK / "task_21_sinusoidal_position",
    BLOCK / "task_22_rope_position",
    BLOCK / "task_23_causal_attention",
    BLOCK / "task_27_minimind_core",
    BLOCK / "task_28_next_token_training",
    BLOCK / "task_29_generate_sampling",
    BLOCK / "task_30_kv_cache",
)
for folder in PATHS:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from generate import generate as generate_without_cache, sample_next_token  # noqa: E402
from kv_cache import (  # noqa: E402
    cache_equivalence_error,
    generate_with_kv_cache,
    logits_with_kv_cache,
    prefill,
)
from mha import MultiHeadSelfAttention  # noqa: E402
from minimind_core import MiniMindConfig, MiniMindCore, RMSNorm  # noqa: E402
from position import sinusoidal_position_encoding  # noqa: E402
from rope import build_rope_cache  # noqa: E402
from train import (  # noqa: E402
    DEFAULT_CORPUS,
    evaluate,
    load_checkpoint,
    make_dataloaders,
    save_checkpoint,
)


def tiny_model(max_seq_len=16):
    torch.manual_seed(7)
    config = MiniMindConfig(
        vocab_size=32,
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=64,
        max_seq_len=max_seq_len,
        pad_token_id=0,
    )
    return MiniMindCore(config).eval()


class Block3Tests(unittest.TestCase):
    def test_position_encodings_validate_dimensions_and_use_distinct_frequencies(self):
        encoding = sinusoidal_position_encoding(6, 8)
        self.assertEqual(encoding.shape, (6, 8))
        self.assertTrue(torch.equal(encoding[0, 0::2], torch.zeros(4)))
        self.assertTrue(torch.equal(encoding[0, 1::2], torch.ones(4)))
        with self.assertRaisesRegex(ValueError, "even"):
            sinusoidal_position_encoding(6, 7)

        cos, sin = build_rope_cache(6, 8)
        self.assertEqual(cos.shape, (6, 4))
        self.assertEqual(sin.shape, (6, 4))
        self.assertFalse(torch.allclose(cos[1, 0], cos[1, 1]))

    def test_task23_really_uses_gqa_rope_and_a_causal_mask(self):
        torch.manual_seed(0)
        attention = MultiHeadSelfAttention(32, num_heads=4, num_kv_heads=2).eval()
        self.assertEqual(attention.q_proj.out_features, 32)
        self.assertEqual(attention.k_proj.out_features, 16)
        self.assertEqual(attention.v_proj.out_features, 16)

        original = torch.randn(1, 6, 32)
        future_changed = original.clone()
        future_changed[:, 4:] = torch.randn_like(future_changed[:, 4:])
        first = attention(original)
        second = attention(future_changed)
        torch.testing.assert_close(first[:, :4], second[:, :4], atol=0, rtol=0)

    def test_core_is_causal_but_last_token_reads_its_prefix(self):
        model = tiny_model()
        original = torch.tensor([[1, 2, 3, 4, 5]])
        future_changed = torch.tensor([[1, 2, 3, 9, 8]])
        original_logits, _ = model(original)
        future_logits, _ = model(future_changed)
        torch.testing.assert_close(
            original_logits[:, :3], future_logits[:, :3], atol=0, rtol=0
        )

        prefix_changed = torch.tensor([[1, 11, 12, 4, 5]])
        prefix_logits, _ = model(prefix_changed)
        difference = (original_logits[:, -1] - prefix_logits[:, -1]).abs().max()
        self.assertGreater(
            difference.item(), 1e-5, "last-position logits must depend on earlier tokens"
        )

    def test_weight_tying_padding_loss_and_cache_shapes(self):
        model = tiny_model()
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        labels = torch.tensor([[2, 3, 4, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        logits, loss = model(input_ids, labels, attention_mask=mask)
        expected = F.cross_entropy(
            logits[:, :3].reshape(-1, 32), labels[:, :3].reshape(-1)
        )
        torch.testing.assert_close(loss, expected)

        _, all_pad_loss = model(
            torch.zeros((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
            attention_mask=torch.zeros((1, 3), dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(all_pad_loss))
        self.assertEqual(all_pad_loss.item(), 0.0)

        _, cache = prefill(model, torch.tensor([[1, 2, 3, 4, 5]]))
        self.assertEqual(len(cache), model.config.n_layers)
        for key, value in cache:
            expected_shape = (1, model.config.n_kv_heads, 5, 8)
            self.assertEqual(key.shape, expected_shape)
            self.assertEqual(value.shape, expected_shape)

    def test_cached_logits_and_greedy_generation_equal_full_recomputation(self):
        model = tiny_model(max_seq_len=10)
        input_ids = torch.tensor([[1, 2, 7, 3, 5]])
        full_logits, _ = model(input_ids)
        cached_logits = logits_with_kv_cache(model, input_ids)
        torch.testing.assert_close(cached_logits, full_logits, atol=1e-6, rtol=1e-5)
        self.assertLess(cache_equivalence_error(model, input_ids), 1e-6)

        ordinary = model.generate(input_ids, max_new_tokens=8, temperature=0)
        cached = generate_with_kv_cache(
            model, input_ids, max_new_tokens=8, temperature=0
        )
        # The eight new tokens force one sliding-window cache rebuild.
        self.assertTrue(torch.equal(cached, ordinary))

    def test_padded_generation_uses_the_mask_and_survives_window_rollover(self):
        torch.manual_seed(13)
        model = MiniMindCore(
            MiniMindConfig(
                vocab_size=32,
                dim=16,
                n_layers=2,
                n_heads=2,
                n_kv_heads=1,
                hidden_dim=32,
                max_seq_len=6,
                pad_token_id=0,
            )
        ).eval()
        left_padded = torch.tensor([[0, 0, 1, 2, 3]])
        mask = left_padded.ne(0)
        explicit = model.generate(
            left_padded, 7, temperature=0, attention_mask=mask
        )
        default = model.generate(left_padded, 7, temperature=0)
        cached = generate_with_kv_cache(
            model, left_padded, 7, temperature=0
        )
        torch.testing.assert_close(default, explicit, atol=0, rtol=0)
        torch.testing.assert_close(cached, explicit, atol=0, rtol=0)

        right_padded = torch.tensor([[1, 2, 3, 0]])
        with self.assertRaisesRegex(ValueError, "left-pad"):
            model.generate(right_padded, 1, temperature=0)
        with self.assertRaisesRegex(ValueError, "left-pad"):
            generate_with_kv_cache(model, right_padded, 1, temperature=0)

    def test_finished_batch_rows_stay_at_eos(self):
        class Config:
            max_seq_len = 8
            pad_token_id = None

        class ScheduledModel:
            config = Config()

            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids, attention_mask=None):
                del attention_mask
                logits = torch.full((*input_ids.shape, 4), -100.0)
                if self.calls == 0:
                    logits[0, -1, 3] = 100.0  # row 0 finishes first
                    logits[1, -1, 1] = 100.0
                else:
                    logits[0, -1, 2] = 100.0  # must be replaced by EOS
                    logits[1, -1, 3] = 100.0
                self.calls += 1
                return logits, None

        generated = generate_without_cache(
            ScheduledModel(),
            torch.tensor([[1], [1]]),
            max_new_tokens=4,
            temperature=0,
            eos_token_id=3,
        )
        self.assertEqual(generated.tolist(), [[1, 3, 3], [1, 1, 3]])

    def test_sampling_greedy_and_top_one_are_argmax(self):
        logits = torch.tensor([[0.1, 2.0, -1.0, 0.3]])
        self.assertEqual(sample_next_token(logits, temperature=0).item(), 1)
        generator = torch.Generator().manual_seed(3)
        selected = sample_next_token(
            logits, temperature=0.8, top_k=1, generator=generator
        )
        self.assertEqual(selected.item(), 1)
        with self.assertRaises(ValueError):
            sample_next_token(logits, top_k=0)
        with self.assertRaises(ValueError):
            sample_next_token(logits, top_p=0)
        with self.assertRaises(ValueError):
            sample_next_token(logits, top_p=1.01)

        # Approximate probabilities [0.60, 0.25, 0.10, 0.05]. With p=0.7,
        # nucleus sampling keeps only the first two candidates.
        nucleus_logits = torch.log(torch.tensor([[0.60, 0.25, 0.10, 0.05]]))
        observed = set()
        generator = torch.Generator().manual_seed(17)
        for _ in range(100):
            observed.add(
                sample_next_token(
                    nucleus_logits, top_p=0.7, generator=generator
                ).item()
            )
        self.assertEqual(observed, {0, 1})

    def test_text_splits_are_independent_and_checkpoint_round_trip_is_exact(self):
        tokenizer, train_loader, val_loader = make_dataloaders(
            DEFAULT_CORPUS, seq_len=24, batch_size=2, seed=0
        )
        self.assertNotEqual(
            train_loader.dataset.ids.data_ptr(), val_loader.dataset.ids.data_ptr()
        )
        shared_length = min(len(train_loader.dataset.ids), len(val_loader.dataset.ids))
        self.assertFalse(
            torch.equal(
                train_loader.dataset.ids[:shared_length],
                val_loader.dataset.ids[:shared_length],
            )
        )

        config = MiniMindConfig(
            vocab_size=tokenizer.vocab_size,
            dim=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            hidden_dim=32,
            max_seq_len=24,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = MiniMindCore(config).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sample_ids, _, sample_mask = next(iter(train_loader))
        before, _ = model(sample_ids, attention_mask=sample_mask)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "round_trip.pt"
            save_checkpoint(
                checkpoint_path, model, optimizer, tokenizer, step=3, val_loss=1.25
            )
            restored, restored_tokenizer, metadata = load_checkpoint(checkpoint_path)
        restored.eval()
        after, _ = restored(sample_ids, attention_mask=sample_mask)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        self.assertIs(restored.lm_head.weight, restored.token_embedding.weight)
        self.assertEqual(restored_tokenizer.token_to_id, tokenizer.token_to_id)
        self.assertEqual(metadata["step"], 3)

    def test_validation_loss_is_weighted_by_valid_token_count(self):
        model = tiny_model(max_seq_len=4)
        input_ids = torch.tensor(
            [[1, 2, 3, 4], [4, 3, 2, 1], [1, 0, 0, 0]]
        )
        labels = torch.tensor(
            [[2, 3, 4, 5], [3, 2, 1, 5], [11, 0, 0, 0]]
        )
        mask = input_ids.ne(0)
        loader = DataLoader(
            TensorDataset(input_ids, labels, mask), batch_size=2, shuffle=False
        )
        reported = evaluate(model, loader)
        with torch.no_grad():
            logits, _ = model(input_ids, labels, attention_mask=mask)
            valid = mask & labels.ne(0)
            expected = F.cross_entropy(logits[valid], labels[valid]).item()
        self.assertAlmostEqual(reported, expected, places=6)

    def test_rmsnorm_preserves_low_precision_activation_dtype(self):
        norm = RMSNorm(8)
        for dtype in (torch.float16, torch.bfloat16):
            values = torch.randn(2, 3, 8).to(dtype=dtype)
            self.assertEqual(norm(values).dtype, dtype)
        double_norm = RMSNorm(8).double()
        double_values = torch.randn(2, 3, 8, dtype=torch.float64)
        expected = double_values * torch.rsqrt(
            double_values.square().mean(dim=-1, keepdim=True) + double_norm.eps
        )
        torch.testing.assert_close(
            double_norm(double_values), expected, atol=1e-12, rtol=1e-12
        )

    def test_attention_updates_and_a_single_batch_can_be_overfit(self):
        torch.manual_seed(2)
        model = MiniMindCore(
            MiniMindConfig(
                vocab_size=12,
                dim=16,
                n_layers=1,
                n_heads=2,
                n_kv_heads=1,
                hidden_dim=32,
                max_seq_len=8,
                pad_token_id=0,
            )
        )
        input_ids = torch.tensor([[1, 2, 3, 4]]).repeat(4, 1)
        labels = torch.tensor([[2, 3, 4, 5]]).repeat(4, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)
        with torch.no_grad():
            _, initial = model(input_ids, labels)

        for step in range(45):
            _, loss = model(input_ids, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                gradient = model.blocks[0].attn.q_proj.weight.grad
                self.assertIsNotNone(gradient)
                self.assertGreater(gradient.abs().sum().item(), 0)
            optimizer.step()

        with torch.no_grad():
            _, final = model(input_ids, labels)
        self.assertLess(final.item(), initial.item() * 0.25)


if __name__ == "__main__":
    unittest.main()
