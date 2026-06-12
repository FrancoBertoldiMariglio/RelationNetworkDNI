# RelationNetworkDNI

A **Relation Network** implemented in PyTorch for binary classification of Argentine ID documents (DNI): deciding whether a document image is **valid or invalid**.

Relation Networks ([Sung et al., 2018](https://arxiv.org/abs/1711.06025)) learn a deep distance metric between a query image and a small support set, which makes them well suited for problems with **limited labeled data** — like fraud/validity detection on documents, where invalid samples are scarce.

## Architecture

- `EmbeddingNet` — convolutional encoder that maps images to feature embeddings
- `RelationModule` — learns a relation score between query embeddings and class prototypes
- Episodic training (n-shot, n-query) with custom weight initialization and StepLR scheduling
- Evaluation with precision / recall / F1 (scikit-learn)

## Usage

```bash
pip install -r requirements.txt

python main.py \
  --valid_dir path/to/valid \
  --invalid_dir path/to/invalid \
  --epochs 50 --n_shot 5 --n_query 10 --learning_rate 1e-3
```

Images are expected in two folders (`valid/`, `invalid/`); the dataset class builds binary labels from the directory structure.

> Note: the dataset of ID documents is private and not included in this repository.
