"""BRIGHT dataset for pre-event RGB + post-event SAR fusion (PaddlePaddle version)."""

import os

import numpy as np
import paddle
import paddle.io
import rasterio
from pycocotools.coco import COCO


class BRIGHTDataset(paddle.io.Dataset):
    """Dataset for BRIGHT building damage instance segmentation (fusion mode).

    Each sample concatenates pre-event RGB (3 bands) and post-event SAR (1 band)
    into a 4-channel tensor [R, G, B, SAR].
    """

    def __init__(
        self,
        ann_file: str,
        image_dir: str,
        pre_event_dir: str,
        transforms=None,
    ):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.pre_event_dir = pre_event_dir
        self.transforms = transforms

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._validate_image_pairs()

    def _validate_image_pairs(self, max_examples: int = 5) -> None:
        missing = []
        for img_id in self.ids:
            fname = self.coco.imgs[img_id]["file_name"]
            post_path = os.path.join(self.image_dir, fname)
            pre_fname = fname.replace("_post_disaster.tif", "_pre_disaster.tif")
            pre_path = os.path.join(self.pre_event_dir, pre_fname)

            missing_parts = []
            if not os.path.isfile(post_path):
                missing_parts.append(f"post-event SAR: {post_path}")
            if not os.path.isfile(pre_path):
                missing_parts.append(f"pre-event RGB: {pre_path}")

            if missing_parts:
                missing.append((fname, missing_parts))

        if not missing:
            return

        example_lines = []
        for file_name, missing_parts in missing[:max_examples]:
            example_lines.append(f"  - {file_name}: {', '.join(missing_parts)}")

        remaining = len(missing) - len(example_lines)
        if remaining > 0:
            example_lines.append(f"  ... and {remaining} more samples")

        raise FileNotFoundError(
            f"Found {len(missing)} samples in {self.ann_file} with missing input files.\n"
            "BRIGHTDataset requires both post-event SAR and pre-event RGB for every annotation entry.\n"
            "Fix the dataset layout or regenerate annotations for the available subset.\n"
            + "\n".join(example_lines)
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple:
        coco = self.coco
        img_id = self.ids[idx]
        img_info = coco.imgs[img_id]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.image_dir, file_name)
        with rasterio.open(img_path) as src:
            sar = src.read(1).astype(np.float32) / 255.0

        pre_fname = file_name.replace("_post_disaster.tif", "_pre_disaster.tif")
        pre_path = os.path.join(self.pre_event_dir, pre_fname)
        with rasterio.open(pre_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0

        image_np = np.concatenate([rgb, sar[np.newaxis]], axis=0)
        image = paddle.to_tensor(image_np, dtype="float32")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            mask = coco.annToMask(ann)
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            masks.append(mask)
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) > 0:
            target = {
                "boxes": paddle.to_tensor(np.array(boxes, dtype=np.float32), dtype="float32"),
                "labels": paddle.to_tensor(np.array(labels, dtype=np.int64), dtype="int64"),
                "masks": paddle.to_tensor(np.array(masks, dtype=np.uint8), dtype="uint8"),
                "image_id": img_id,
                "area": paddle.to_tensor(np.array(areas, dtype=np.float32), dtype="float32"),
                "iscrowd": paddle.to_tensor(np.array(iscrowd, dtype=np.int64), dtype="int64"),
            }
        else:
            h_img, w_img = image_np.shape[-2], image_np.shape[-1]
            target = {
                "boxes": paddle.zeros([0, 4], dtype="float32"),
                "labels": paddle.zeros([0], dtype="int64"),
                "masks": paddle.zeros([0, h_img, w_img], dtype="uint8"),
                "image_id": img_id,
                "area": paddle.zeros([0], dtype="float32"),
                "iscrowd": paddle.zeros([0], dtype="int64"),
            }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class RandomVerticalFlip:
    """Randomly flip image and target vertically."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = paddle.flip(image, [-2])
            height = image.shape[-2]

            if target["boxes"].numel() > 0:
                boxes = target["boxes"]
                boxes_flipped = boxes.clone()
                boxes_flipped[:, 1] = height - boxes[:, 3]
                boxes_flipped[:, 3] = height - boxes[:, 1]
                target["boxes"] = boxes_flipped

            if target["masks"].numel() > 0:
                target["masks"] = paddle.flip(target["masks"], [-2])

        return image, target


class RandomHorizontalFlip:
    """Randomly flip image and target horizontally."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = paddle.flip(image, [-1])
            width = image.shape[-1]

            if target["boxes"].numel() > 0:
                boxes = target["boxes"]
                boxes_flipped = boxes.clone()
                boxes_flipped[:, 0] = width - boxes[:, 2]
                boxes_flipped[:, 2] = width - boxes[:, 0]
                target["boxes"] = boxes_flipped

            if target["masks"].numel() > 0:
                target["masks"] = paddle.flip(target["masks"], [-1])

        return image, target


class Compose:
    """Compose transforms that operate on (image, target) pairs."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transforms(train: bool):
    """Return transforms for training or validation."""
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms)
