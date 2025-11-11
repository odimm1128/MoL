# MoL

This repository provides a Mixture-of-Layers (MoL) experiment for
image classification on MNIST. It freezes a pretrained MobileNetV3-Large
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

- `--top-k`: number of MobileNet feature stages to select per sample (default: 2)
- `--feature-dim`: dimension of the projected layer features (default: 256)
- `--num-workers`: DataLoader worker processes for MNIST I/O (default: 4)
- `--learning-curve-path`: save a train/validation loss & accuracy plot (requires matplotlib)
- `--final-checkpoint-path`: always save the last-epoch weights for later inference
- `--save-path`: optional checkpoint path for the best validation model

See `python train_mol_mnist.py --help` for the full list of options.

## Running inference

Use the saved checkpoint with the inference script:

```bash
python run_mol_mnist.py --checkpoint path/to/checkpoint.pth --preview-samples 5
```

This downloads the MNIST test split if needed, reports loss/accuracy, and prints a few sample predictions. Adjust `--batch-size`, `--num-workers`, or `--preview-samples` as desired.
