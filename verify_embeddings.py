import torch
from transformers import AutoModel


original = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-text-v1-unsupervised",
    trust_remote_code=True
)


modified = AutoModel.from_pretrained(
    "biomedical-nomic-surgery",
    trust_remote_code=True
)


old_weights = original.embeddings.word_embeddings.weight.data[:30528]

new_weights = modified.embeddings.word_embeddings.weight.data[:30528]


difference = torch.abs(old_weights - new_weights)


print("Maximum difference:")
print(difference.max().item())


print("Average difference:")
print(difference.mean().item())


print(
    "Embeddings identical:",
    torch.allclose(
        old_weights,
        new_weights,
        atol=1e-6
    )
)