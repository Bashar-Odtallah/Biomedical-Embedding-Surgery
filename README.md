# Biomedical Embedding Surgery

Manual embedding surgery performed on:

nomic-ai/nomic-embed-text-v1-unsupervised

The goal was to expand the model vocabulary with biomedical terminology while keeping all original embeddings unchanged.

## Overview

The original tokenizer splits specialized biomedical terms into multiple subword fragments.

Example:

Before surgery:

glioblastoma
→ g, ##lio, ##bla, ##sto, ##ma

After surgery:

glioblastoma
→ glioblastoma


15 biomedical tokens were added:

- glioblastoma
- immunohistochemistry
- pharmacogenomics
- angiogenesis
- neurodegeneration
- bioavailability
- oncogenesis
- metastasis
- cytokinemia
- microvasculature
- electrophysiology
- histopathology
- chemoresistance
- osteogenesis
- hepatocarcinogenesis


## Method

1. Extract the original embedding matrix.
2. Generate new embeddings by averaging the embeddings of the original subword pieces.
3. Expand the embedding matrix manually.
4. Insert the new vectors into unused vocabulary positions.
5. Modify the tokenizer vocabulary manually.
6. Verify that old embeddings remain unchanged.


## Verification

### Embedding verification

The original vocabulary embeddings were compared against the modified model.

Result:

Maximum difference: 0.0

Average difference: 0.0

Original embeddings unchanged.


### Tokenizer verification

The added biomedical terms are now represented as single tokens.

Example:

glioblastoma
→ [30522]


### End-to-end verification

Biomedical text successfully passes through:

Tokenizer → Model → Embedding output

Output shape:

(batch, sequence length, 768)


## Files

| File | Purpose |
|---|---|
| manual_embedding_surgery.py | Creates new biomedical embeddings |
| modify_tokenizer.py | Adds biomedical tokens into main vocabulary |
| verify_embeddings.py | Checks original embeddings remain unchanged |
| verify_tokenizer.py | Checks new token IDs |
| verify_end_to_end.py | Tests complete pipeline |
| biomedical-nomic-surgery/ | Final modified model |


## Running

Install dependencies:

pip install -r requirements.txt


Run surgery:

python manual_embedding_surgery.py

python modify_tokenizer.py


Verify:

python verify_embeddings.py

python verify_tokenizer.py

python verify_end_to_end.py


## Model

The final model is uploaded to Hugging Face Hub.