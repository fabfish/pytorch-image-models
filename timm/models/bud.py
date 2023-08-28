""" DeiT - Data-efficient Image Transformers

DeiT model defs and weights from https://github.com/facebookresearch/deit, original copyright below

paper: `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

paper: `DeiT III: Revenge of the ViT` - https://arxiv.org/abs/2204.07118

Modifications copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial
from typing import Sequence, Union

import torch
from torch import nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer, trunc_normal_, checkpoint_filter_fn
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['BudgetedVisionTransformerDistilled']  # model_registry will add each entrypoint fn to this

from .vision_transformer import Mlp, LayerScale, DropPath, Block, Attention # BudgetedAttention
from torch.nn import functional as F
from torch.jit import Final
from timm.layers import use_fused_attn

def generate_gaussian_mask(size, sigma):
    center = size // 2
    x, y = torch.meshgrid(torch.arange(size), torch.arange(size))
    mask = torch.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    mask = mask / mask.max()  # Normalize to [0, 1]
    return mask

def generate_batch_masks(batch_size, size, sigma):
    probability_masks = torch.stack([generate_gaussian_mask(size, sigma) for _ in range(batch_size)])
    binary_masks = torch.bernoulli(probability_masks)

    return probability_masks, binary_masks


class BudgetedAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            mlp_ratio=4.,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        # print(dim, num_heads, mlp_ratio)
        self.mlp_ratio = int(mlp_ratio)
        self.num_heads = num_heads
        self.head_dim = int(dim * (mlp_ratio / 4) // num_heads)
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, int(dim * (mlp_ratio / 4) * 3), bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(int(dim * (mlp_ratio / 4)) , dim)
        self.proj_drop = nn.Dropout(proj_drop)

        

    def forward(self, x):
        # B, N, C = x.shape
        B, N, C = x.shape
        C = int(C * (self.mlp_ratio / 4))
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BudgetedBlock(nn.Module):
    '''
    Budgeted Block
    '''

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = BudgetedAttention(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.mlp_ratio = mlp_ratio
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.sigma = sigma

    def forward(self, x):
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # 创建一个 sigma 字典，sigma = 3, 5 对应 mlp_ratio = 1, 2
        mlp_sigma_dict = {1: 3, 2: 5, 4: 100}

        if self.training and self.mlp_ratio !=4:
            if self.mlp_ratio in mlp_sigma_dict.keys():
                sigma = mlp_sigma_dict[self.mlp_ratio]
            else:
                sigma = 100

            # print(x.shape)
            attn_scores = self.attn(self.norm1(x))
            # print(attn_scores.shape) # 128, 197, 384
            batch_size = x.shape[0]
            size = 14
            _, binary_masks = generate_batch_masks(batch_size, size, sigma)
            binary_masks = binary_masks.view(batch_size,-1).to(x.device)

            # class token
            new_row = torch.ones(batch_size, attn_scores.shape[1] - size*size).to(x.device)
            binary_masks = torch.cat([new_row, binary_masks], dim=1)

            masked_attn_scores = attn_scores * binary_masks.unsqueeze(-1)
            
            x = x + self.drop_path1(self.ls1(masked_attn_scores))

        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BudgetedVisionTransformerDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1),
            x),
            dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def forward_features(self, x) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1),
            x),
            dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        if pre_logits:
            return (x + x_dist) / 2
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def _create_bud(variant, pretrained=False, distilled=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model_cls = BudgetedVisionTransformerDistilled if distilled else VisionTransformer
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # deit models (FB weights)
    'bud_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'bud_small_patch16_224_baby': _cfg(),
    'bud_small_patch16_224_child': _cfg(),
    'bud_small_patch16_224_man': _cfg(),
})

@register_model
def bud_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn = Block, )
    model = _create_bud('bud_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def bud_small_patch16_224_baby(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=2, mlp_ratio = 1, block_fn = BudgetedBlock, )
    model = _create_bud('bud_small_patch16_224_baby', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def bud_small_patch16_224_child(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=4, mlp_ratio = 2, block_fn = BudgetedBlock, )
    model = _create_bud('bud_small_patch16_224_child', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def bud_small_patch16_224_man(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio = 4, block_fn = BudgetedBlock, )
    model = _create_bud('bud_small_patch16_224_man', pretrained=pretrained, **dict(model_args, **kwargs))
    return model