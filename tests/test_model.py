from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch

from cyrus.config import load_config
from cyrus.model import CyrusModel, ModelHyperparameters, choose_device


ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        torch.manual_seed(4256)
        self.hyperparameters = ModelHyperparameters(
            vocab_size=96,
            context_length=32,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            dropout=0.0,
        )

    def test_shape_loss_weight_tying_and_backward(self) -> None:
        model = CyrusModel(self.hyperparameters)
        inputs = torch.randint(0, self.hyperparameters.vocab_size, (2, 16))
        targets = torch.randint(0, self.hyperparameters.vocab_size, (2, 16))
        logits, loss = model(inputs, targets)
        self.assertEqual(tuple(logits.shape), (2, 16, self.hyperparameters.vocab_size))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertEqual(model.token_embedding.weight.data_ptr(), model.lm_head.weight.data_ptr())

    def test_causal_mask_prevents_future_information_leakage(self) -> None:
        model = CyrusModel(self.hyperparameters).eval()
        left = torch.randint(0, self.hyperparameters.vocab_size, (1, 16))
        right = left.clone()
        right[:, 8:] = torch.randint(0, self.hyperparameters.vocab_size, (1, 8))
        with torch.no_grad():
            left_logits, _ = model(left)
            right_logits, _ = model(right)
        torch.testing.assert_close(left_logits[:, :8], right_logits[:, :8], rtol=0, atol=1e-6)

    def test_state_dict_round_trip_and_seeded_generation_are_deterministic(self) -> None:
        model = CyrusModel(self.hyperparameters).eval()
        clone = CyrusModel(self.hyperparameters).eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()))
        prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        first = model.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=20, seed=99)
        second = clone.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=20, seed=99)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_tiny_batch_can_be_deliberately_overfit(self) -> None:
        config = ModelHyperparameters(
            vocab_size=32,
            context_length=16,
            d_model=48,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            dropout=0.0,
        )
        model = CyrusModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * 2)
        targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]] * 2)
        with torch.no_grad():
            _, initial = model(inputs, targets)
        assert initial is not None
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            assert loss is not None
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            _, final = model(inputs, targets)
        assert final is not None
        self.assertLess(final.item(), initial.item() * 0.25)

    def test_smoke_profile_is_within_owner_parameter_budget(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml")
        hyperparameters = ModelHyperparameters(
            vocab_size=config.tokenizer.vocab_size,
            context_length=config.model.context_length,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            n_kv_heads=config.model.n_kv_heads,
            dropout=config.model.dropout,
            rope_theta=config.model.rope_theta,
        )
        model = CyrusModel(hyperparameters)
        self.assertGreaterEqual(model.parameter_count, 1_000_000)
        self.assertLessEqual(model.parameter_count, 5_000_000)
        self.assertEqual(choose_device("auto").type, "cpu")


if __name__ == "__main__":
    unittest.main()

