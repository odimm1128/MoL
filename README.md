# MoL

This repository provides a Mixture-of-Layers (MoL) experiment for
image classification on MNIST. It freezes a pretrained ResNet-18
backbone and learns a lightweight gating mechanism that routes a small
number of intermediate layer features through a trainable three-layer
MLP classifier.

## Training the model

Install the dependencies and run the training script:

```bash
pip install torch torchvision
python train_mol_mnist.py --epochs 10 --batch-size 128
```

Key arguments:

- `--top-k`: number of ResNet layers to select per sample (default: 2)
- `--feature-dim`: dimension of the projected layer features (default: 256)
- `--save-path`: optional checkpoint path for the best validation model

See `python train_mol_mnist.py --help` for the full list of options.
