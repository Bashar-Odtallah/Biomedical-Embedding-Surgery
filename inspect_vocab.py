import json

with open("original_tokenizer/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

print(tokenizer_data.keys())

print()

print(tokenizer_data["model"].keys())