"""Training entry-point for BRIGHT building damage instance segmentation (PaddlePaddle version).

Usage::

    python -m src.train --config config/disaster.yaml
    python -m src.train --config config/disaster.yaml --epochs 5
    python -m src.train --config config/disaster.yaml --resume outputs/latest.pdparams
"""

import argparse
import logging
import os

import paddle
from paddle.io import DataLoader

from src.utils import load_config, set_seed, collate_fn
from src.dataset.data import BRIGHTDataset, get_transforms
from src.model.mask_rcnn import build_model
from src.engine import train_one_epoch, evaluate, format_metrics_report


def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on BRIGHT data (PaddlePaddle)")
    parser.add_argument("--config", default="config/disaster.yaml", help="Path to YAML config file")
    parser.add_argument("--resume", default="", help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(log_file, mode="a"))

    set_seed(train_cfg["seed"])

    place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
    paddle.set_device(place)
    logger.info(f"Using device: {'GPU' if paddle.is_compiled_with_cuda() else 'CPU'}")

    image_dir = os.path.join(data_cfg["root"], data_cfg["images_dir"])
    pre_event_dir = os.path.join(data_cfg["root"], data_cfg["pre_event_dir"])

    train_dataset = BRIGHTDataset(
        ann_file=data_cfg["train_ann"],
        image_dir=image_dir,
        pre_event_dir=pre_event_dir,
        transforms=get_transforms(train=True),
    )

    val_dataset = BRIGHTDataset(
        ann_file=data_cfg["val_ann"],
        image_dir=image_dir,
        pre_event_dir=pre_event_dir,
        transforms=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        use_shared_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        use_shared_memory=True,
    )

    model = build_model(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        pixel_mean=data_cfg["pixel_mean"],
        pixel_std=data_cfg["pixel_std"],
        box_detections_per_img=model_cfg.get("box_detections_per_img", 1500),
        rpn_pre_nms_top_n_test=model_cfg.get("rpn_pre_nms_top_n_test", 1500),
        rpn_post_nms_top_n_test=model_cfg.get("rpn_post_nms_top_n_test", 1500),
    )

    total_params = 0
    trainable_params = 0
    frozen_params = 0
    components: dict = {}

    for name, param in model.named_parameters():
        n = np.prod(param.shape)
        total_params += n
        parts = name.split(".")
        comp = parts[0]
        if len(parts) > 1:
            comp = f"{comp}.{parts[1]}"
        if comp == "backbone.body" and len(parts) > 2:
            comp = f"{comp}.{parts[2]}"

        if comp not in components:
            components[comp] = {"trainable": 0, "frozen": 0}
        if not param.stop_gradient:
            trainable_params += n
            components[comp]["trainable"] += n
        else:
            frozen_params += n
            components[comp]["frozen"] += n

    logger.info("Parameter summary:")
    logger.info(f"  Total: {total_params:,}  |  Trainable: {trainable_params:,}  |  Frozen: {frozen_params:,}")
    for comp, counts in sorted(components.items()):
        total_c = counts["trainable"] + counts["frozen"]
        status = "FROZEN" if counts["trainable"] == 0 else ("TRAINABLE" if counts["frozen"] == 0 else "MIXED")
        logger.info(f"  {comp:<30s} {total_c:>10,} params  [{status}]")

    params_list = [p for p in model.parameters() if not p.stop_gradient]
    optimizer = paddle.optimizer.Momentum(
        parameters=params_list,
        learning_rate=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )

    lr_scheduler = paddle.optimizer.lr.MultiStepDecay(
        learning_rate=train_cfg["lr"],
        milestones=train_cfg["lr_steps"],
        gamma=train_cfg["lr_gamma"],
    )

    start_epoch = 0
    best_ap = 0.0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        state_dict = paddle.load(args.resume)
        if isinstance(state_dict, dict) and "model" in state_dict:
            model.set_state_dict(state_dict["model"])
            optimizer.set_state_dict(state_dict["optimizer"])
            start_epoch = state_dict["epoch"] + 1
            best_ap = state_dict.get("best_ap", 0.0)
        else:
            model.set_state_dict(state_dict)
        logger.info(f"  Resumed at epoch {start_epoch}, best segm_AP so far: {best_ap:.4f}")

    for epoch in range(start_epoch, epochs):
        lr_scheduler.step(epoch)

        train_one_epoch(
            model,
            optimizer,
            train_loader,
            epoch,
            print_freq=train_cfg["print_freq"],
            warmup_iters=train_cfg.get("warmup_iters", 0),
            warmup_factor=train_cfg.get("warmup_factor", 0.001),
            log_file=log_file,
        )

        eval_dir = os.path.join(output_dir, "eval")
        metrics = evaluate(model, val_loader, output_dir=eval_dir, epoch=epoch, log_file=log_file)
        logger.info(f"Epoch [{epoch}] evaluation: {metrics}")
        for line in format_metrics_report(metrics):
            logger.info(f"Epoch [{epoch}] {line}")

        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "best_ap": best_ap,
        }

        latest_path = os.path.join(output_dir, "latest.pdparams")
        paddle.save(checkpoint_data, latest_path)
        logger.info(f"  Saved latest checkpoint: {latest_path}")

        segm_ap = metrics.get("segm_AP", 0.0)
        if segm_ap > best_ap:
            best_ap = segm_ap
            checkpoint_data["best_ap"] = best_ap
            best_path = os.path.join(output_dir, "best_model.pdparams")
            paddle.save(checkpoint_data, best_path)
            logger.info(f"  New best segm_AP: {best_ap:.4f} -- saved to {best_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
