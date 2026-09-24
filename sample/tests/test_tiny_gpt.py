import unittest

import torch
from torch import nn
from torch.nn import functional as F

from tiny_gpt.config import Config
from tiny_gpt.dataset import encode_documents, make_batch
from tiny_gpt.generate import generate
from tiny_gpt.model import LayerNorm, TinyGPT, scaled_attention
from tiny_gpt.tokenizer import Tokenizer


class TinyGPTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.config = Config()
        self.config.d_model = 16
        self.config.context_length = 8
        self.config.batch_size = 2
        self.tokenizer = Tokenizer.from_texts(
            ["猫は眠っています。", "犬も眠っています。", "猫と犬です。"]
        )

    def test_tokenizer_round_trip(self) -> None:
        text = "犬は眠っています。"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_tokenizer_replaces_unseen_characters(self) -> None:
        ids = self.tokenizer.encode("猫と🦉")
        self.assertEqual(ids[-1], self.tokenizer.unk_id)
        self.assertEqual(self.tokenizer.decode(ids), "猫と<unk>")

    def test_documents_have_separate_end_tokens(self) -> None:
        texts = ["猫です。", "犬です。"]
        ids = encode_documents(texts, self.tokenizer).tolist()
        self.assertEqual(ids.count(self.tokenizer.eos_id), 2)
        boundary = ids.index(self.tokenizer.eos_id)
        self.assertEqual(self.tokenizer.decode(ids[:boundary]), texts[0])
        self.assertEqual(self.tokenizer.decode(ids[boundary + 1:-1]), texts[1])

    def test_batch_targets_are_shifted_one_token(self) -> None:
        data = torch.arange(40)
        generator = torch.Generator().manual_seed(42)
        x, y = make_batch(data, self.config, generator, torch.device("cpu"))
        self.assertEqual(x.shape, (2, 8))
        torch.testing.assert_close(y, x + 1)

    def test_causal_attention_and_value_mixing(self) -> None:
        q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        k = q.clone()
        v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
        output, weights = scaled_attention(q, k, v)
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 2))
        self.assertEqual(weights[0, 0, 1].item(), 0.0)
        torch.testing.assert_close(output[0, 0], v[0, 0])
        doubled, same_weights = scaled_attention(q, k, v * 2)
        torch.testing.assert_close(weights, same_weights)
        torch.testing.assert_close(doubled, output * 2)

    def test_layer_norm_forward_and_gradient(self) -> None:
        ours = LayerNorm(4)
        reference = nn.LayerNorm(4, eps=ours.epsilon)
        first = torch.randn(2, 3, 4, requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        torch.testing.assert_close(ours(first), reference(second))
        weights = torch.randn(2, 3, 4)
        (ours(first) * weights).sum().backward()
        (reference(second) * weights).sum().backward()
        torch.testing.assert_close(first.grad, second.grad, atol=0.00001, rtol=0.00001)
        torch.testing.assert_close(ours.scale.grad, reference.weight.grad)

    def test_future_tokens_do_not_change_earlier_predictions(self) -> None:
        model = TinyGPT(self.tokenizer.vocab_size, self.config)
        first = torch.tensor([[3, 5, 8, 2]])
        second = torch.tensor([[3, 5, 9, 4]])
        torch.testing.assert_close(model(first)[:, :2], model(second)[:, :2])

    def test_training_reduces_loss_on_a_small_pattern(self) -> None:
        model = TinyGPT(self.tokenizer.vocab_size, self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]])
        y = torch.tensor([2, 1, 2, 1, 2, 1, 2, 1])
        with torch.no_grad():
            initial = F.cross_entropy(model(x).reshape(8, -1), y).item()
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).reshape(8, -1), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = F.cross_entropy(model(x).reshape(8, -1), y).item()
        self.assertLess(final, initial * 0.25)

    def test_generation_seed_and_context_limit(self) -> None:
        model = TinyGPT(self.tokenizer.vocab_size, self.config)
        device = torch.device("cpu")
        first = generate(model, self.tokenizer, "猫は", 12, device, seed=5)
        second = generate(model, self.tokenizer, "猫は", 12, device, seed=5)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("猫は"))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 9, dtype=torch.long))

    def test_generation_stops_at_document_end(self) -> None:
        model = TinyGPT(self.tokenizer.vocab_size, self.config)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias.zero_()
            model.lm_head.bias[self.tokenizer.eos_id] = 100
        output = generate(model, self.tokenizer, "猫は", 12, torch.device("cpu"), method="greedy")
        self.assertEqual(output, "猫は")


if __name__ == "__main__":
    unittest.main()
