import json


path = "biomedical-nomic-surgery/tokenizer.json"


with open(path, "r", encoding="utf-8") as f:
    tokenizer = json.load(f)


vocab = tokenizer["model"]["vocab"]


new_tokens = [
    "glioblastoma",
    "immunohistochemistry",
    "pharmacogenomics",
    "angiogenesis",
    "neurodegeneration",
    "bioavailability",
    "oncogenesis",
    "metastasis",
    "cytokinemia",
    "microvasculature",
    "electrophysiology",
    "histopathology",
    "chemoresistance",
    "osteogenesis",
    "hepatocarcinogenesis"
]


start_id = max(vocab.values()) + 1


for i, token in enumerate(new_tokens):
    vocab[token] = start_id + i


with open(path, "w", encoding="utf-8") as f:
    json.dump(tokenizer, f, indent=2)


print("New vocabulary size:", len(vocab))
print("Added IDs:")
print(start_id, "-", start_id + len(new_tokens)-1)