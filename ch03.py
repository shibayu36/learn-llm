import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# journeyの、それぞれの内積が出ているだけ
print(attn_scores_2)

attn_weights = torch.softmax(attn_scores_2, dim=0)
print("Attention weights: ", attn_weights)
print("Sum: ", attn_weights.sum())

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights[i] * x_i

print("Context vector: ", context_vec_2)

attn_scores = inputs @ inputs.T

print("Attention scores: ", attn_scores)

all_context_vecs = attn_scores @ inputs
print("All context vectors: ", all_context_vecs)

x_2 = inputs[1]
d_in = inputs.shape[1] # 入力埋め込みのサイズ。3
d_out = 2 # 出力埋め込みのサイズ。2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
