from transformers import AutoTokenizer


model_path = "biomedical-nomic-surgery"


# Load modified tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)


print("Tokenizer size:")
print(len(tokenizer))

print()


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


for word in domain_tokens:

    ids = tokenizer(
        word,
        add_special_tokens=False
    )["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("Word:")
    print(word)

    print("IDs:")
    print(ids)

    print("Tokens:")
    print(tokens)

    print("Single token:")
    print(len(ids) == 1)

    print("-" * 40)