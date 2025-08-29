# Hinglish → English Neural Machine Translation

A Transformer-based Neural Machine Translation (NMT) model for translating Hinglish (Hindi-English mixed text) into English.

---

## Features

- Transformer architecture for sequence-to-sequence translation
- Preprocessing and tokenization of Hinglish text
- Trained model included (tracked with Git LFS)
- Jupyter Notebook for training and inference

---

## Dataset

The dataset used for training is hosted externally due to size limitations:

[Download Dataset](https://huggingface.co/datasets/nateraw/english-to-hinglish/viewer/default/train?views%5B%5D=train)

> Please download the dataset before running the notebook or scripts.

---

## Model

The trained Transformer model (`transformer_model.pth`) is tracked using Git LFS. Ensure Git LFS is installed:

```bash
git lfs install
