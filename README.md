# MoL

This repository provides a Mixture-of-Layers (MoL) experiment for
image classification on MNIST. It freezes a pretrained MobileNetV3-Large
backbone and learns a lightweight gating mechanism that routes a small
number of intermediate layer features through a trainable three-layer
MLP classifier.

## Training on MNIST

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

## WM-811K wafer maps

The repository also includes `wm811k_mol.py`, which adds a `train`/`run` CLI for the [WM-811K wafer map dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/data). Place `LSWMD.pkl` (or the original `.zip`) under `data/wm811k/` or pass the file path directly via `--data-root`. Install the additional dependencies:

```bash
pip install pandas pillow
```

Example sanity sweep (50 samples per class, 20% validation split, one epoch):

```bash
python wm811k_mol.py train \
  --data-root data/LSWMD.pkl \
  --epochs 1 \
  --batch-size 32 \
  --limit-per-class 50 \
  --val-split 0.2 \
  --learning-curve-path data/wm811k_curve.png \
  --save-path data/wm811k_best.pth \
  --final-checkpoint-path data/wm811k_final.pth
```

To evaluate a checkpoint and print routed-layer previews:

```bash
python wm811k_mol.py run \
  --checkpoint data/wm811k_best.pth \
  --data-root data/LSWMD.pkl \
  --preview-samples 10
```

This command also produces `wm811k_predictions.csv` (override with `--predictions-csv`) capturing every wafer's `id,target,pred,layers` so you can filter or diff runs later.

Notable WM-811K options:

- `--limit-per-class`: cap the number of wafers per failure type for quick experiments.
- `--val-split`: fractional split carved from the filtered dataset (default 0.1).
- `--no-class-weights`: disable inverse-frequency weighting if you prefer raw CE loss.
- `--image-size`: resize target fed into the frozen MobileNet backbone (default 224).
- `--predictions-csv`: path for the per-sample inference CSV.
- Unlabeled wafers tagged as `none` remain in the label space for inference, but the training split automatically drops them so they don't influence the weight updates.

### Capturing inference logs

The `run` subcommand now reports the WM-811K confusion matrix, gate distribution, and a routed-layer preview in a single pass. Pipe the command to a log file if you want to track experiments:

```powershell
python wm811k_mol.py run `
  --checkpoint data/wm811k_best.pth `
  --data-root data/LSWMD.pkl `
  --preview-samples 20 `
  2>&1 | Tee-Object infer_log.txt
```

On bash/zsh you can use `python ... | tee infer_log.txt` instead. The resulting `infer_log.txt` keeps the evaluation summary alongside the printed confusion matrix so you can compare runs without rerunning inference.
