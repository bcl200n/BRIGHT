"""Training and evaluation engine (PaddlePaddle version)."""

import json
import os
import tempfile
import warnings

import numpy as np
import paddle

try:
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

from src.utils import MetricLogger, SmoothedValue


EVAL_IOU_TYPES = ("bbox", "segm")
EVAL_MAX_DETS = (1, 100, 1500)


def _sanitize_category_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def build_empty_coco_metrics(coco_gt, iou_types=EVAL_IOU_TYPES, max_dets=EVAL_MAX_DETS) -> dict:
    metrics = {}
    cat_ids = list(coco_gt.getCatIds())

    for iou_type in iou_types:
        metrics.update({
            f"{iou_type}_AP": 0.0,
            f"{iou_type}_AP50": 0.0,
            f"{iou_type}_AP75": 0.0,
            f"{iou_type}_APs": 0.0,
            f"{iou_type}_APm": 0.0,
            f"{iou_type}_APl": 0.0,
            f"{iou_type}_AR{max_dets[0]}": 0.0,
            f"{iou_type}_AR{max_dets[1]}": 0.0,
            f"{iou_type}_AR{max_dets[2]}": 0.0,
            f"{iou_type}_ARs": 0.0,
            f"{iou_type}_ARm": 0.0,
            f"{iou_type}_ARl": 0.0,
        })

        for cat_id in cat_ids:
            cat_name = coco_gt.cats[cat_id]["name"]
            metrics[f"{iou_type}_AP_{_sanitize_category_name(cat_name)}"] = 0.0

    return metrics


def summarize_coco_eval(coco_eval, iou_type: str) -> dict:
    stats = coco_eval.stats
    max_dets = coco_eval.params.maxDets
    metrics = {
        f"{iou_type}_AP": float(stats[0]),
        f"{iou_type}_AP50": float(stats[1]),
        f"{iou_type}_AP75": float(stats[2]),
        f"{iou_type}_APs": float(stats[3]),
        f"{iou_type}_APm": float(stats[4]),
        f"{iou_type}_APl": float(stats[5]),
        f"{iou_type}_AR{max_dets[0]}": float(stats[6]),
        f"{iou_type}_AR{max_dets[1]}": float(stats[7]),
        f"{iou_type}_AR{max_dets[2]}": float(stats[8]),
        f"{iou_type}_ARs": float(stats[9]),
        f"{iou_type}_ARm": float(stats[10]),
        f"{iou_type}_ARl": float(stats[11]),
    }

    precisions = coco_eval.eval.get("precision")
    if precisions is None:
        return metrics

    for cat_idx, cat_id in enumerate(coco_eval.params.catIds):
        precision = precisions[:, :, cat_idx, 0, -1]
        precision = precision[precision > -1]
        cat_name = coco_eval.cocoGt.cats[cat_id]["name"]
        metrics[f"{iou_type}_AP_{_sanitize_category_name(cat_name)}"] = (
            float(precision.mean()) if precision.size else 0.0
        )

    return metrics


def format_metrics_report(metrics: dict) -> list:
    lines = []

    for iou_type in EVAL_IOU_TYPES:
        prefix = f"{iou_type}_"
        if f"{prefix}AP" not in metrics:
            continue

        lines.append(
            f"{iou_type}: "
            f"AP={metrics[f'{prefix}AP']:.4f}  "
            f"AP50={metrics[f'{prefix}AP50']:.4f}  "
            f"AP75={metrics[f'{prefix}AP75']:.4f}  "
            f"APs={metrics[f'{prefix}APs']:.4f}  "
            f"APm={metrics[f'{prefix}APm']:.4f}  "
            f"APl={metrics[f'{prefix}APl']:.4f}"
        )

        ar_keys = [key for key in metrics if key.startswith(f"{prefix}AR") and key[7:].isdigit()]
        if ar_keys:
            ar_keys = sorted(ar_keys, key=lambda key: int(key[7:]))
            ar_parts = [f"{key[5:]}={metrics[key]:.4f}" for key in ar_keys]
            lines.append(f"{iou_type} recall: " + "  ".join(ar_parts))

        class_keys = [key for key in metrics if key.startswith(f"{prefix}AP_")]
        if class_keys:
            class_parts = []
            for key in sorted(class_keys):
                class_name = key[len(f"{prefix}AP_"):]
                class_parts.append(f"{class_name}={metrics[key]:.4f}")
            lines.append(f"{iou_type} per-class AP: " + "  ".join(class_parts))

    return lines


def evaluate_coco_results(
    coco_gt,
    results_path: str,
    iou_types=EVAL_IOU_TYPES,
    max_dets=EVAL_MAX_DETS,
) -> dict:
    coco_dt = coco_gt.loadRes(results_path)
    metrics = {}

    for iou_type in iou_types:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.params.maxDets = list(max_dets)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics.update(summarize_coco_eval(coco_eval, iou_type))

    return metrics


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    epoch,
    print_freq=10,
    warmup_iters=0,
    warmup_factor=0.001,
    log_file=None,
):
    """Run one full training epoch (PaddlePaddle version)."""
    model.train()
    metric_logger = MetricLogger(delimiter="  ", log_file=log_file)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    warmup_current_factor = warmup_factor
    if epoch == 0 and warmup_iters > 0:
        warmup_iters = min(warmup_iters, len(data_loader) - 1)
    else:
        warmup_iters = 0

    base_lrs = [opt.get("learning_rate", opt.get("lr", 0.02)) for opt in optimizer._param_groups]

    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if warmup_iters > 0 and i < warmup_iters:
            warmup_factor_i = warmup_factor + (1.0 - warmup_factor) * (i / warmup_iters)
            for param_group, base_lr in zip(optimizer._param_groups, base_lrs):
                param_group["learning_rate"] = base_lr * warmup_factor_i

        images = [paddle.to_tensor(img) if not isinstance(img, paddle.Tensor) else img for img in images]
        targets = [
            {k: (paddle.to_tensor(v) if isinstance(v, np.ndarray) else v)
             for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = float(losses.numpy())

        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            raise SystemExit(1)

        losses.backward()
        optimizer.step()
        optimizer.clear_grad()

        metric_logger.update(loss=loss_value, **{k: float(v.numpy()) for k, v in loss_dict.items()})
        metric_logger.update(lr=optimizer._param_groups[0]["learning_rate"])

    return metric_logger


def evaluate(model, data_loader, output_dir=None, epoch=None, log_file=None):
    """Run COCO-style evaluation for bbox and segm AP (PaddlePaddle version)."""
    import logging
    _logger = logging.getLogger("train")
    if not _logger.handlers:
        _logger.addHandler(logging.StreamHandler())
    if log_file and not any(isinstance(h, logging.FileHandler) for h in _logger.handlers):
        _logger.addHandler(logging.FileHandler(log_file, mode="a"))

    model.eval()
    coco_gt = data_loader.dataset.coco

    coco_results = []
    num_images = len(data_loader)

    for idx, (images, targets) in enumerate(data_loader):
        images = [paddle.to_tensor(img) if not isinstance(img, paddle.Tensor) else img for img in images]

        with paddle.no_grad():
            outputs = model(images)

        for target, output in zip(targets, outputs):
            img_id = target["image_id"]
            image_id = int(img_id.numpy()) if isinstance(img_id, paddle.Tensor) else int(img_id)

            boxes = output["boxes"].numpy()
            scores = output["scores"].numpy()
            labels = output["labels"].numpy()
            masks = output["masks"].numpy()

            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i].tolist()
                bbox_coco = [x1, y1, x2 - x1, y2 - y1]

                mask_bin = (masks[i, 0] > 0.5).astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(mask_bin))
                rle["counts"] = rle["counts"].decode("utf-8")

                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": bbox_coco,
                    "score": float(scores[i]),
                    "segmentation": rle,
                })

        if (idx + 1) % 50 == 0 or (idx + 1) == num_images:
            _logger.info(f"  Eval: [{idx + 1}/{num_images}] {len(coco_results)} detections")

    if len(coco_results) == 0:
        _logger.info("No predictions produced -- returning zero metrics.")
        return build_empty_coco_metrics(coco_gt)

    cleanup_results_path = False
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        tag = f"_epoch{epoch:03d}" if epoch is not None else ""
        results_path = os.path.join(output_dir, f"eval_results{tag}.json")
        with open(results_path, "w") as f:
            json.dump(coco_results, f)
        _logger.info(f"  Saved {len(coco_results)} eval predictions to {results_path}")
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(coco_results, f)
            results_path = f.name
            cleanup_results_path = True

    try:
        return evaluate_coco_results(coco_gt, results_path)
    finally:
        if cleanup_results_path and os.path.exists(results_path):
            os.remove(results_path)
