import torch
import json
from transformers import AutoModel, AutoTokenizer
import os
print("CURRENT DIRECTORY:")
print(os.getcwd())


model_name = "nomic-ai/nomic-embed-text-v1-unsupervised"


# load model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True
)


embedding_layer = model.embeddings.word_embeddings


old_embeddings = embedding_layer.weight.data.clone()


print("Original embedding size:", old_embeddings.shape)


domain_tokens = [
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


# create embeddings for newer tokens

new_vectors = []


for word in domain_tokens:

    ids = tokenizer(
        word,
        add_special_tokens=False
    )["input_ids"]

    pieces = tokenizer.convert_ids_to_tokens(ids)

    vectors = old_embeddings[ids]

    new_vector = vectors.mean(dim=0)

    new_vectors.append(new_vector)

    print(word)
    print("Pieces:", pieces)
    print("New embedding created")
    print()


new_vectors = torch.stack(new_vectors)


print("New vectors:", new_vectors.shape)


# build expanded embedding matrix

required_size = 30592


new_embedding_matrix = torch.zeros(
    required_size,
    old_embeddings.shape[1]
)


# copy old embeddings before modification

new_embedding_matrix[:old_embeddings.shape[0]] = old_embeddings


# insert new biomedical embedding

new_embedding_matrix[
    30522:30537
] = new_vectors


# replace model embedding

model.embeddings.word_embeddings.weight.data = new_embedding_matrix


print(
    "New embedding size:",
    model.embeddings.word_embeddings.weight.shape
)


# verifying it

unchanged = torch.equal(
    old_embeddings,
    model.embeddings.word_embeddings.weight[:old_embeddings.shape[0]]
)


print(
    "Old embeddings unchanged:",
    unchanged
)

# Update config to match new size
model.config.vocab_size = required_size

# Save
print("\nCONFIG:")
print(model.config)

print("\nVOCAB SIZE:", model.config.vocab_size)

print("\nEmbedding shape:")
print(model.embeddings.word_embeddings.weight.shape)

print("Tokenizer size:", len(tokenizer))

# save

save_path = r"C:\Users\HomePC\Desktop\Biomedical-Embedding-Surgery\biomedical-nomic-surgery"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


# manually updating config file

config_path = f"{save_path}/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

config["vocab_size"] = required_size

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)


print("Config updated!")
print("Saved vocab size:", config["vocab_size"])

print("Model saved!")