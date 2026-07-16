from __future__ import annotations

import torch

from barellm.sampling.sampler import (
    greedy,
    sample_temperature,
    sample_top_k,
    sample_top_p,
)


class TestGreedy:
    def test_returns_argmax(self) -> None:
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
        assert greedy(logits).item() == 1

    def test_returns_tensor(self) -> None:
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
        result = greedy(logits)
        assert isinstance(result, torch.Tensor)

    def test_ties_pick_first(self) -> None:
        logits = torch.tensor([5.0, 5.0, 1.0])
        assert greedy(logits).item() == 0


class TestSampleTemperature:
    def test_near_zero_is_greedy(self) -> None:
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 5.0, 2.0])
        assert sample_temperature(logits, temperature=0.001).item() == 1

    def test_returns_tensor(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(50)
        result = sample_temperature(logits, temperature=0.7)
        assert isinstance(result, torch.Tensor)

    def test_high_temp_can_pick_anything(self) -> None:
        torch.manual_seed(0)
        logits = torch.tensor([0.01, 0.01, 0.01, 0.01])
        result = sample_temperature(logits, temperature=100.0)
        assert 0 <= result.item() < 4

    def test_valid_index(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(1000)
        result = sample_temperature(logits, temperature=0.7)
        assert 0 <= result.item() < 1000


class TestSampleTopK:
    def test_k_one_is_greedy(self) -> None:
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
        assert sample_top_k(logits, k=1).item() == 1

    def test_only_picks_from_top_k(self) -> None:
        torch.manual_seed(42)
        logits = torch.tensor([10.0, 9.0, -100.0, -100.0])
        for _ in range(20):
            result = sample_top_k(logits, k=2).item()
            assert result in (0, 1)

    def test_returns_tensor(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(100)
        result = sample_top_k(logits, k=10)
        assert isinstance(result, torch.Tensor)

    def test_valid_index(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(1000)
        result = sample_top_k(logits, k=20)
        assert 0 <= result.item() < 1000


#
class TestSampleTopP:
    def test_p_one_keeps_all(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(100)
        result = sample_top_p(logits, p=1.0)
        assert 0 <= result.item() < 100

    def test_extreme_confidence_keeps_one(self) -> None:
        torch.manual_seed(42)
        logits = torch.tensor([100.0, 1.0, 1.0, 1.0])
        for _ in range(20):
            result = sample_top_p(logits, p=0.9).item()
            assert result == 0

    def test_returns_tensor(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(100)
        result = sample_top_p(logits, p=0.9)
        assert isinstance(result, torch.Tensor)

    def test_valid_index(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(1000)
        result = sample_top_p(logits, p=0.9)
        assert 0 <= result.item() < 1000
