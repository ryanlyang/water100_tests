import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.datasets as datasets
from torchvision import transforms


class FoodSubset(datasets.ImageFolder):

    def __init__(
            self,
            root,
            split='train',
            cfg=None,
            transform=None,
            target_transform=None,
            is_valid_file=None,
    ):
        self.cfg = cfg
        self.split = split
        self.food_subset_dir = getattr(self.cfg.DATA, "FOOD_SUBSET_DIR", "food-101")
        self.dataset_root = os.path.join(root, self.food_subset_dir)
        self.size = int(cfg.DATA.SIZE)
        self.remove_background = bool(getattr(cfg.DATA, "REMOVE_BACKGROUND", False))
        self.return_attention = cfg.DATA.ATTENTION_DIR != "NONE"
        self.return_seg = self._should_return_seg()
        self.segmentation_dir = self._get_segmentation_dir()

        super().__init__(
            os.path.join(self.dataset_root, "train"),
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )

        if self.cfg.DATA.CLASSES is None:
            self.classes = []
            with open(os.path.join(self.dataset_root, "meta", "labels.txt")) as f:
                for line in f:
                    self.classes.append(
                        line.replace('\n', '').replace(' ', '_').lower()
                    )
        else:
            self.classes = sorted(self.cfg.DATA.CLASSES)

        metadata_path = self._resolve_metadata_path()
        df = pd.read_csv(metadata_path)
        split_col = getattr(self.cfg.DATA, "SPLIT", "split")
        if split_col not in df.columns:
            raise KeyError(f"Missing split column '{split_col}' in {metadata_path}")

        df = df[df["label"].astype(str).isin(self.classes)]
        self.df = df[df[split_col].astype(str) == str(split)].copy()
        self.filename_array = self.df["abs_file_path"].astype(str).values
        self.imgs = [
            (os.path.join(self.dataset_root, f), self.classes.index(str(l)))
            for f, l in zip(self.df["abs_file_path"], self.df["label"])
        ]
        self.data = np.array(
            [os.path.join(self.dataset_root, str(f)) for f in df["abs_file_path"]]
        )
        self.samples = self.imgs
        self.groups = [0 for i in range(len(self.imgs))]

        if self.return_attention:
            self.attention_data = np.array(
                [
                    os.path.join(
                        self.dataset_root,
                        cfg.DATA.ATTENTION_DIR,
                        os.path.splitext(path)[0] + '.pth'
                    )
                    for path in self.filename_array
                ]
            )
        else:
            self.attention_data = None

        if self.return_seg:
            self.seg_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
            ])
            self.seg_data = np.array([self._seg_path_for_image(path) for path in self.filename_array])
        else:
            self.seg_transform = None
            self.seg_data = None


    def __getitem__(self, index):
        path, target = self.imgs[index]
        group = self.groups[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_attention:
            att = torch.load(self.attention_data[index])
            if self.cfg.DATA.ATTENTION_DIR == 'deeplabv3_attention':
                att = att['mask']
            else:
                att = att['unnormalized_attentions']
            att = F.interpolate(att, size=(self.size, self.size), mode='bilinear', align_corners=False)
        else:
            att = torch.Tensor([-1])  # So batches correctly

        if self.return_seg:
            seg_path = self.seg_data[index]
            if not os.path.exists(seg_path):
                raise FileNotFoundError(
                    f"Segmentation/mask file not found: {seg_path}\n"
                    f"SEGMENTATION_DIR: {self.segmentation_dir}\n"
                    "Expected masks like '<class>_<image>.png' in prediction_cmap."
                )
            seg = Image.open(seg_path).convert("L")
            seg = self.seg_transform(seg)
            seg = (seg > 0).float()
            if self.remove_background:
                sample = sample * seg
        else:
            seg = torch.Tensor([-1])  # So batches correctly

        null = torch.Tensor([-1])  # So batches correctly

        return {
            'image_path': path,
            'image': sample,
            'label': target,
            'seg': seg,
            'group': group,
            'bbox': null,
            'attention': att,
            'index': index,
            'split': self.split,
        }

    def _resolve_metadata_path(self):
        candidates = [
            os.path.join(self.dataset_root, "meta", "all_images.csv"),
            os.path.join(self.dataset_root, "all_images.csv"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not find all_images.csv. Checked: " + ", ".join(candidates)
        )

    def _loss_uses_gt(self, gt_name):
        if not hasattr(self.cfg, "EXP") or "LOSSES" not in self.cfg.EXP:
            return False
        for loss_name in self.cfg.EXP.LOSSES:
            loss_settings = self.cfg.EXP.LOSSES[loss_name]
            if ("COMPUTE" in loss_settings and loss_settings.COMPUTE) or \
               ("LOG" in loss_settings and loss_settings.LOG):
                if "GT" in loss_settings and loss_settings.GT == gt_name:
                    return True
        return False

    def _should_return_seg(self):
        train_only = bool(getattr(self.cfg.DATA, "SEG_TRAIN_ONLY", False))
        if train_only and self.split != "train":
            return bool(getattr(self.cfg.DATA, "RETURN_SEG", False)) or self.remove_background
        return bool(getattr(self.cfg.DATA, "RETURN_SEG", False)) or \
            self.remove_background or self._loss_uses_gt("segmentation")

    def _get_segmentation_dir(self):
        seg_dir = getattr(self.cfg.DATA, "SEGMENTATION_DIR", None)
        if seg_dir is None:
            seg_dir = "NONE"
        if isinstance(seg_dir, str) and seg_dir.upper() != "NONE":
            seg_dir = os.path.expanduser(seg_dir)
            if not os.path.isabs(seg_dir):
                seg_dir = os.path.join(self.cfg.DATA.ROOT, seg_dir)
            return seg_dir
        return os.path.join(self.dataset_root, "segmentations")

    def _seg_path_for_image(self, img_rel_path):
        # img_rel_path usually looks like "images/<class>/<img>.jpg".
        img_rel_path = img_rel_path.strip().lstrip(os.sep)
        rel_no_ext = os.path.splitext(img_rel_path)[0]
        base = os.path.basename(rel_no_ext)
        parent = os.path.basename(os.path.dirname(rel_no_ext)).replace(".", "_")
        rel_without_images = rel_no_ext[len("images/"):] if rel_no_ext.startswith("images/") else rel_no_ext

        candidates = [
            os.path.join(self.segmentation_dir, f"{parent}_{base}.png"),
            os.path.join(self.segmentation_dir, f"{base}.png"),
            os.path.join(self.segmentation_dir, rel_no_ext + ".png"),
            os.path.join(self.segmentation_dir, rel_without_images + ".png"),
            os.path.join(self.segmentation_dir, parent, f"{base}.png"),
            os.path.join(self.segmentation_dir, rel_without_images + ".jpg"),
            os.path.join(self.segmentation_dir, rel_without_images + ".jpeg"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]
