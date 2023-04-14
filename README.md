# SimRE: Simple Contrastive Learning with Soft Logical Rule for Knowledge Graph Embedding
Official code repository "SimRE: Simple Contrastive Learning with Soft Logical Rule for Knowledge Graph Embedding".

### Requirements

- python>=3.7
- torch>=1.6 (for mixed precision training)
- transformers>=4.15

### How to Run

Step 1: preprocess the dataset

```shell
bash data/preprocess.sh
```

Step 2: mining rule by RNNLogic

https://github.com/DeepGraphLearning/RNNLogic

Step 3, training the model

```
python src/main.py
```

Step 4, evaluate a trained model

```
python src/evaluate.py
```
