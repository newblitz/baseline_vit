# Baseline Diffusion

Train a pretrained TinyViT image classifier from the provided PKL split files.

## Run

```bash
uv run python train_swin_tiny.py --epochs 50
```

The script expects the image paths referenced in `cpsmi2025_train_list.pkl` and `cpsmi2025_test_list.pkl` to exist under the workspace, for example:

- `dataset/cpsmi2025/lesion/low_grade/low_grade128.jpg`
- `dataset/cpsmi2025/normal/squamous/squamous236.jpg`
- `dataset/cpsmi2025/cancer/invasive/invasive254.jpg`
