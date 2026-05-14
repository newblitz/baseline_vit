# Baseline Diffusion

Train a pretrained TinyViT image classifier from the provided PKL split files.

## Run

```bash
uv run python train_swin_tiny.py --epochs 50
```

Default PKLs:

- cpsmi2025_train_list.pkl
- cpsmi2025_test_list.pkl

## Kaggle Path Changes

If your PKL stores paths like dataset/cpsmi2025/... but Kaggle mounts files at /kaggle/input/<dataset-name>/..., use path remapping:

```bash
uv run python train_swin_tiny.py \
  --train-pkl cpsmi2025_train_list.pkl \
  --test-pkl cpsmi2025_test_list.pkl \
  --data-root /kaggle/input \
  --path-replace-from dataset/ \
  --path-replace-to "" \
  --epochs 50
```

If your PKL paths are already absolute or already correct for Kaggle, only set --train-pkl/--test-pkl and optionally --data-root.
