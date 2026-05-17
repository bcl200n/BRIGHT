"""Category definitions for BRIGHT building damage instance segmentation."""

CATEGORIES = {
    1: "intact",
    2: "damaged",
    3: "destroyed",
}

NUM_CLASSES = len(CATEGORIES) + 1
