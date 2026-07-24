import torch
from transformers import AutoModel, AutoTokenizer


path = "biomedical-nomic-surgery"


tokenizer = AutoTokenizer.from_pretrained(
    path,
    trust_remote_code=True
)


model = AutoModel.from_pretrained(
    path,
    trust_remote_code=True
)


text = "Glioblastoma research uses immunohistochemistry techniques."


inputs = tokenizer(
    text,
    return_tensors="pt"
)


with torch.no_grad():
    output = model(**inputs)


print("Input IDs:")
print(inputs["input_ids"])


print("\nToken count:")
print(len(inputs["input_ids"][0]))


print("\nModel output:")
print(output.last_hidden_state.shape)