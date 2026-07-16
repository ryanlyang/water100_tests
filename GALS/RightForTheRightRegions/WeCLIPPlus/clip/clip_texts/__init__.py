"""Dataset-specific CLIP text configurations."""

DATASET_MODULES = {
    "waterbirds": "clip.clip_texts.clip_text_waterbirds",
    "redmeat": "clip.clip_texts.clip_text_redmeat",
    "decoymnist": "clip.clip_texts.clip_text_mnist",
    "coloredmnist": "clip.clip_texts.clip_text_mnist",
}

