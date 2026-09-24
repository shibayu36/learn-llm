import torch
from tiny_gpt.model import scaled_attention

q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])

_, normal_weights = scaled_attention(q, k, v, causal=False)
_, causal_weights = scaled_attention(q, k, v, causal=True)
print(normal_weights)
print(causal_weights)
assert causal_weights[0, 0, 1].item() == 0.0
