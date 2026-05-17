# BRIGHT Challenge (PaddlePaddle Version)

PaddlePaddle rewrite of the Mask R-CNN baseline for the [BRIGHT Challenge](https://chrx97.com/challenge.html) at CVPR 2026 MONTI Workshop.

This is a complete reimplementation of the [original PyTorch baseline](../cvprw26) using PaddlePaddle, preserving the same model architecture, training pipeline, and evaluation protocol.

## Key Changes from PyTorch Version

| PyTorch | PaddlePaddle |
|---------|-------------|
| `torch` / `torchvision` | `paddle` / `paddle.vision` |
| `torch.utils.data.Dataset` | `paddle.io.Dataset` |
| `torch.utils.data.DataLoader` | `paddle.io.DataLoader` |
| `torch.optim.SGD` | `paddle.optimizer.Momentum` |
| `torch.optim.lr_scheduler` | `paddle.optimizer.lr` |
| `torch.amp.autocast` / `GradScaler` | Not used (amp disabled) |
| `torch.save` / `torch.load` | `paddle.save` / `paddle.load` |
| `.pth` checkpoints | `.pdparams` checkpoints |
| `maskrcnn_resnet50_fpn` (torchvision) | `MaskRCNN` (paddle.vision) |

## Setup

```bash
conda create -n bright_paddle python=3.10 -y
conda activate bright_paddle
pip install paddlepaddle-gpu>=2.6.0
pip install -e .
```

## Usage

**Train:**
```bash
python -m src.train --config config/disaster.yaml
# or
bash train.sh
```

**Inference (holdout):**
```bash
python -m src.infer --config config/disaster.yaml
# or
bash infer.sh
```

**Evaluation (server-side):**
```bash
python src/eval.py --gt /path/to/holdout_gt.json --predictions outputs/infer/predictions.json.gz
```

## Dataset Layout

Same as PyTorch version:
```
<BRIGHT_ROOT>/
├── post-event/
├── pre-event/
└── target_instance_level/
```

## Citation

```bibtex
@article{chen2025bright,
  title   = {BRIGHT: A globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response},
  author  = {Chen, Hongruixuan and others},
  journal = {Earth System Science Data},
  volume  = {17},
  pages   = {6217--6243},
  year    = {2025},
  doi     = {10.5194/essd-17-6217-2025}
}
```
