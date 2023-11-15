# Transformers experiments

## Aim of this repository

This repository is to provide a detailed and simple explanation of the transformer architecture. The repository also contains experiments and bits of code for building and training transformer networks.

At the moment the repo contain explanations on vanilla transformers but will include explanations on spin-off transformers such as Vision Transformers.

## Explanations on embeddings and positional encodings

- A [Jupyter notebook](/vanilla_transformers/1-embeddings/embdeddings.ipynb) displaying the workflow for obtaining embeddings from sentences. This includes tokenization of the sentence + getting the positional encodings

- The [README](/vanilla_transformers/1-embeddings/README.md) detailing the theory behind the terms embeddings, tokenization and positional encodings

## Resources to understand transformers

**Seminal papers:**

- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
- [An image is worth 16x16 words](https://arxiv.org/abs/2010.11929)

**Other cool papers:**

- [A survey of transformers](https://www.sciencedirect.com/science/article/pii/S2666651022000146)
- [A Survey of Audio Classification Using Deep Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10258355)

**Pre-trained transformers for acoustic data:**

- [BEATs](https://github.com/microsoft/unilm/tree/master/beats). But better to use [our repository](https://github.com/NINAnor/rare_species_detections) as it contains a running example on ESC50.
- [AST: Audio Spectrogram Transformer](https://github.com/YuanGongND/ast)
- [PaSST: Efficient Training of Audio Transformers with Patchout](https://github.com/kkoutini/PaSST)

**Tutorials related to transformers:**

- [TransformerFromScratch Tutorial Series](https://github.com/Animadversio/TransformerFromScratch)
- [StatQuest on Transformers](https://www.youtube.com/watch?v=zxQyTK8quyY)

**Other cool resources on transformers:**

- [A very simple visual explanations on how LLMs work](https://www.theguardian.com/technology/ng-interactive/2023/nov/01/how-ai-chatbots-like-chatgpt-or-bard-work-visual-explainer)
