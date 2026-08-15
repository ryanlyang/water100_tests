import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_Plus.PAR import PAR

from pretrained.facebookDinov2.hubconf import (
    dinov2_vits14,
    dinov2_vitb14,
    dinov2_vitl14,
    dinov2_vits14_reg,
    dinov2_vitb14_reg,
    dinov2_vitl14_reg,
)


_DINOV2_MODELS = {
    "dinov2_vits14": dinov2_vits14,
    "dinov2_vitb14": dinov2_vitb14,
    "dinov2_vitl14": dinov2_vitl14,
    "dinov2_vits14_reg": dinov2_vits14_reg,
    "dinov2_vitb14_reg": dinov2_vitb14_reg,
    "dinov2_vitl14_reg": dinov2_vitl14_reg,
}


def _infer_patch_size(model_name: str, default: int = 14) -> int:
    m = re.search(r"_p(\d+)", str(model_name).lower())
    if m is not None:
        return int(m.group(1))
    if "14" in str(model_name):
        return 14
    if "16" in str(model_name):
        return 16
    return int(default)


def _load_dino_like_encoder(model_name: str):
    if model_name in _DINOV2_MODELS:
        return _DINOV2_MODELS[model_name](pretrained=True)

    if "xcit" in str(model_name).lower():
        try:
            import timm
        except Exception as exc:
            raise ImportError(
                "XCiT backbone requested but timm is not available. "
                "Install timm in the WeCLIP environment."
            ) from exc

        candidates = [str(model_name)]
        if not re.search(r"_(224|384)$", str(model_name)):
            candidates.append(f"{model_name}_224")

        last_exc = None
        for candidate in candidates:
            try:
                return timm.create_model(candidate, pretrained=True, num_classes=0)
            except TypeError:
                # Some timm versions do not accept num_classes for this call signature.
                return timm.create_model(candidate, pretrained=True)
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            f"Could not create XCiT model '{model_name}' via timm. Tried: {candidates}"
        ) from last_exc

    raise ValueError(
        f"Unknown DINO/XCiT model: {model_name}. "
        f"Supported DINOv2: {sorted(_DINOV2_MODELS.keys())}. "
        "XCiT is supported via timm model names (e.g., xcit_medium_24_p16)."
    )


def _to_patch_tokens_and_hw(feat: torch.Tensor, batch_size: int, default_h: int, default_w: int, source: str):
    """Convert feature output into [B, N, C] patch tokens and infer (H, W) token grid."""
    if feat.ndim == 4:
        # [B, C, H, W] -> [B, N, C]
        b, c, h, w = feat.shape
        if b != batch_size:
            raise RuntimeError(f"{source}: batch mismatch in 4D tensor ({b} vs {batch_size})")
        tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return tokens, h, w

    if feat.ndim != 3:
        raise RuntimeError(f"{source}: unsupported feature rank {feat.ndim}, expected 3D/4D.")

    if feat.shape[0] != batch_size:
        raise RuntimeError(f"{source}: batch mismatch in 3D tensor ({feat.shape[0]} vs {batch_size})")

    # Prefer [B, N, C]. If [B, C, N], transpose.
    expected_tokens = default_h * default_w
    if feat.shape[1] == expected_tokens:
        tokens = feat
    elif feat.shape[2] == expected_tokens:
        tokens = feat.permute(0, 2, 1).contiguous()
    else:
        tokens = feat
        n = tokens.shape[1]
        # Strip a small number of prefix tokens (e.g., cls/register tokens).
        if n > expected_tokens and (n - expected_tokens) <= 8:
            tokens = tokens[:, n - expected_tokens :, :]

    n_tokens = tokens.shape[1]
    if n_tokens == expected_tokens:
        return tokens, default_h, default_w

    # Fallback: infer token grid from token count.
    if default_h > 0 and n_tokens % default_h == 0:
        return tokens, default_h, n_tokens // default_h
    if default_w > 0 and n_tokens % default_w == 0:
        return tokens, n_tokens // default_w, default_w

    side = int(round(float(n_tokens) ** 0.5))
    if side * side == n_tokens:
        return tokens, side, side

    raise RuntimeError(
        f"{source}: cannot infer token grid from n_tokens={n_tokens} "
        f"(default grid={default_h}x{default_w})."
    )

def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)
    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class WeCLIP_Plus(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, dino_model=None, dino_fts_dim=768, decoder_layers=3,
                 embedding_dim=256, in_channels=512, dataset_root_path=None, clip_flag=16, device='cuda',
                 clip_pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dino_fts_fuse_dim = dino_fts_dim #384 for vit-s, 768for vit-b
        self.clip_flag = clip_flag
        self.dino_model_name = str(dino_model)
        self.dino_patch_size = _infer_patch_size(self.dino_model_name, default=14)

        self.encoder, _ = clip.load(clip_model, device=device, pretrained=clip_pretrained)

        for name, param in self.encoder.named_parameters():
            if clip_flag == 14 and '23' not in name:
                param.requires_grad=False
            if clip_flag == 16 and "11" not in name:
                 param.requires_grad=False

        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)


        self.dino_encoder = _load_dino_like_encoder(self.dino_model_name)


        for name, param in self.dino_encoder.named_parameters():
            param.requires_grad = False

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1) #index=11
        
        self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers, heads=8, output_dim=self.num_classes)


        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'],
                                               self.encoder)  # ['a rendering of a weird {}.'], model) (20, 512)


        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]

        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform, clip_flag=clip_flag)
        
        self.root_path = os.path.join(dataset_root_path, 'JPEGImages')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda() #1,2,4,8,12,24
        self.iter_num = 0
        self.require_all_fts = True


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        for param in list(self.dino_decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups
    


    def forward(
        self,
        img,
        img_names='2007_000032',
        mode='train',
        return_cam_labels=True,
    ):
        all_img_tokens_list = []
        cam_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True, clip_flag=self.clip_flag)

        with torch.no_grad():
            dino_img_h, dino_img_w = (
                (h // self.dino_patch_size) * self.dino_patch_size,
                (w // self.dino_patch_size) * self.dino_patch_size,
            )
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)
            dino_ftses = self.dino_encoder.forward_features(dino_img)
            if isinstance(dino_ftses, dict):
                dino_fts = None
                for key in ('x_norm_patchtokens', 'x_patchtokens', 'x_prenorm', 'x'):
                    if key in dino_ftses:
                        dino_fts = dino_ftses[key]
                        break
                if dino_fts is None:
                    raise RuntimeError(
                        f"Unsupported feature dict keys from '{self.dino_model_name}': "
                        f"{sorted(dino_ftses.keys())}"
                    )
            else:
                dino_fts = dino_ftses

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)

        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-2].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//self.clip_flag, w //self.clip_flag) #(11, b, c, h, w)
        all_img_tokens = all_img_tokens[-1].unsqueeze(0)

        fts = self.decoder_fts_fuse(all_img_tokens)
        _, _, fts_h, fts_w = fts.shape #24

        dino_grid_h = max(dino_img_h // self.dino_patch_size, 1)
        dino_grid_w = max(dino_img_w // self.dino_patch_size, 1)

        if isinstance(dino_fts, list):
            for d_i, dino_fts_single in enumerate(dino_fts):
                dino_fts_single, grid_h, grid_w = _to_patch_tokens_and_hw(
                    dino_fts_single,
                    b,
                    dino_grid_h,
                    dino_grid_w,
                    f"{self.dino_model_name}[{d_i}]",
                )
                dino_fts_single = dino_fts_single.reshape([b, grid_h, grid_w, -1]).permute(0, 3, 1, 2)
                dino_fts[d_i] = dino_fts_single

            dino_fts = torch.stack(dino_fts)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts)
            dino_h, dino_w = grid_h, grid_w

        else:
            dino_fts, grid_h, grid_w = _to_patch_tokens_and_hw(
                dino_fts,
                b,
                dino_grid_h,
                dino_grid_w,
                self.dino_model_name,
            )
            dino_fts = dino_fts.reshape([b, grid_h, grid_w, -1]).permute(0,3,1,2)
            _, _, dino_h, dino_w = dino_fts.shape #32
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
        
        dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)

        seg_clip, seg_attn_weight_list_clip = self.decoder(fts)
        seg_dino, seg_attn_weight_list_dino = self.decoder(dino_fts)

        clip_dino_fts = torch.cat([fts, dino_fts], dim=1)

        seg_dino_prob = F.softmax(0.5*seg_dino+0.5*seg_clip, dim=1)
        seg_dino_prob = seg_dino_prob.detach()

        attn_fts = F.interpolate(clip_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        # Teacher-map export consumes only the fused segmentation logits. PAR
        # labels are an expensive training-side auxiliary output, particularly
        # when VOC XML metadata refers to large original ImageNet dimensions.
        if not return_cam_labels:
            return seg_clip, seg_dino, None, attn_pred

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.jpg')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]

            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]

            require_seg_trans = True
            seg_dino_cam = seg_dino_prob[i]

            # iter_w = 0.5

            cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   self.bg_text_features, self.fg_text_features,
                                                                   self.grad_cam,
                                                                   mode=mode,
                                                                   require_seg_trans=require_seg_trans,
                                                                   seg_dino_cam=seg_dino_cam,
                                                                   clip_flag = self.clip_flag
                                                                          )

            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()

            cams = torch.cat([bg_score, cams], dim=0).cuda()
            
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()
            
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            
            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)

        if self.training:
            return seg_clip, seg_dino, all_cam_labels, attn_pred
        else:
            return seg_clip, seg_dino, all_cam_labels, attn_pred

        
    

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeCLIP_Plus('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)
