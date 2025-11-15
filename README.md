# MoL

Mixture-of-Layers (MoL) is a lightweight classifier that freezes a pretrained
backbone, projects a handful of intermediate layers, and lets a learned gating
network route only the most useful stages through a compact MLP head. This
repository contains a flavour:

1. **WM-811K wafer maps (SwiftFormer backbone)** – a higher-capacity variant
   that swaps the backbone for a pretrained SwiftFormer from `timm` and reports
   per-class confusion matrices.

## WM-811K wafer maps (SwiftFormer backbone)

`wm811k_mol.py` exposes `train` / `run` subcommands geared towards the
[WM-811K wafer map dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/data).
The script instantiates a pretrained SwiftFormer backbone via `timm`, freezes it,
and layers the MoL gating head on top.

1. Place `LSWMD.pkl` (or the original `.zip`) under `data/` or pass the
   file path directly via `--data-root`.
2. Install the extra dependencies (adds Google Drive downloads for the pretrained SwiftFormer weights):

   ```bash
   pip install pandas pillow timm einops gdown
   ```

3. Train (example: 50 wafers per class, 20% validation, SwiftFormer-S backbone):

   ```bash
   python wm811k_mol.py train \
     --data-root data/LSWMD.pkl \
     --epochs 1 \
     --batch-size 32 \
     --limit-per-class 50 \
     --val-split 0.2 \
     --backbone swiftformer_s \
     --learning-curve-path data/wm811k_curve.png \
     --save-path data/wm811k_best.pth \
     --final-checkpoint-path data/wm811k_final.pth
   ```

4. Evaluate a checkpoint and emit routed-layer previews + confusion matrix:

   ```bash
   python wm811k_mol.py run \
     --checkpoint data/wm811k_best.pth \
     --data-root data/LSWMD.pkl \
     --preview-samples 10
   ```

During training the script reports class distributions, gate usage, and (if
requested) learning curves. The `run` subcommand prints the WM-811K confusion
matrix, gate distribution, and per-sample routed layers, and also writes
`wm811k_predictions.csv` (override with `--predictions-csv`) containing
`id,target,pred,layers` for the entire evaluation set.

During the first run the script automatically downloads the official ImageNet
pretrained SwiftFormer weights from the authors' Google Drive links into
`./pretrained/swiftformer` (override via `--swiftformer-cache-dir`). These
weights seed the frozen backbone, while the MoL gating head is still trained
on WM-811K from scratch.

Notable WM-811K options:

- `--swiftformer-cache-dir`: location to store/download the SwiftFormer weights.
- `--limit-per-class`: cap the number of wafers per failure type for quick experiments.
- `--val-split`: fractional split carved from the filtered dataset (default 0.1).
- `--no-class-weights`: disable inverse-frequency weighting if you prefer raw CE loss.
- `--image-size`: resize target fed into the frozen backbone (default 224).
- `--predictions-csv`: path for the per-sample inference CSV.
- Unlabeled wafers tagged as `none` remain in the label space for inference, but the
  training split automatically drops them so they don't influence the weight updates.
