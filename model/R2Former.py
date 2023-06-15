# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed, Block
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import time
# import cv2


class R2Former(nn.Module):
    def __init__(self, img_size=(224,224), decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
                 decoder_mlp_ratio=4., decoder_norm_layer=nn.LayerNorm, embed_dim=384, num_classes=1, num_patches=16, input_dim=256, num_corr=1):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.num_patches = num_patches
        self.num_corr = num_corr
        self.img_size = img_size
        self.patch_size = [16, 16]
        # self.patch_embed.num_patches
        # MAE decoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_2, std=.02)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=True)
        self.ratio = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=decoder_norm_layer)
            for i in range(decoder_depth)])

        self.blocks_2 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=decoder_norm_layer)
            for i in range(2)])

        # self.local_dim = 128
        self.decoder_norm = decoder_norm_layer(decoder_embed_dim)
        # self.global_head = nn.Linear(self.input_dim, decoder_embed_dim, bias=True)
        # self.local_head = nn.Linear(self.local_dim, decoder_embed_dim, bias=True)
        self.pair_head = nn.Linear(7, decoder_embed_dim, bias=True)
        self.pair_head_2 = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        # self.pair_head = nn.Sequential(nn.Linear(7 * self.num_corr, decoder_embed_dim, bias=False),
        #                                 nn.BatchNorm1d(decoder_embed_dim),
                                        # nn.ReLU(inplace=True), # hidden layer
                                        # nn.Linear(decoder_embed_dim, decoder_embed_dim)) # output layer

        self.decoder_pred = nn.Linear(decoder_embed_dim, num_classes, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.decoder_pos_embed.data[:, 1:].normal_(mean=0.0, std=0.01)
        self.initialize_weights_all()
        self.cos = nn.CosineSimilarity(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

        # self.decoder_pred.weight.data.normal_(mean=0.0, std=0.01)
        # self.decoder_pred.bias.data.zero_()

    def initialize_weights_all(self):
        # initialization
        # pos_embed = get_2d_sincos_pos_embed_wh(self.decoder_pos_embed.shape[-1], (self.img_size[0]//self.patch_size[0],self.img_size[1]//self.patch_size[1]*2),
        #                                     cls_token=True)
        # pos_embed = torch.from_numpy(pos_embed).float()
        # self.decoder_pos_embed[0, :(1 + self.num_patches // 2)] = pos_embed[pos_embed.shape[0]//2-self.img_size[1]//self.patch_size[1]//2]
        # self.decoder_pos_embed[0, (1 + self.num_patches // 2):] = pos_embed[
        #     pos_embed.shape[0] // 2 + self.img_size[1] // self.patch_size[1] // 2]

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def to_map(self, x):
        p = self.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, x.shape[-1]))
        x = torch.permute(x, (0, 3, 1, 2))
        return x

    def forward(self, x_global, x_rerank=None, y_global=None, y_rerank=None):
        if x_rerank is None and y_global is None and y_rerank is None:
            x_rerank = y_rerank = torch.rand(1,500,131).cuda()
            y_global = x_global = torch.rand(1,256).cuda()
        # print(x_global.shape, x_rerank.shape)
        B = x_global.shape[0]
        N = x_rerank.shape[1]
        global_score = self.cos(x_global.detach(), y_global.detach())

        x_rerank_token = F.normalize(x_rerank[:, :, 3:], p=2, dim=2)
        y_rerank_token = F.normalize(y_rerank[:, :, 3:], p=2, dim=2)
        x_coordinate = x_rerank[:, :, :3].detach().clamp(min=0, max=1)
        y_coordinate = y_rerank[:, :, :3].detach().clamp(min=0, max=1)

        correlation = torch.matmul(x_rerank_token, y_rerank_token.permute((0, 2, 1)))
        xy_matrix = torch.cat(
            [x_coordinate.unsqueeze(2).repeat(1, 1, y_rerank_token.shape[1], 1),
             y_coordinate.unsqueeze(1).repeat(1, x_rerank_token.shape[1], 1, 1), correlation.unsqueeze(3)],
            dim=3)
        order_q = torch.argsort(correlation.unsqueeze(3), dim=2, descending=True).repeat(1, 1, 1, 7)
        order_k = torch.argsort(correlation.unsqueeze(3), dim=1, descending=True).repeat(1, 1, 1, 7)
        select_q = torch.gather(input=xy_matrix, index=order_q[:, :, :self.num_corr, :], dim=2)
        select_k = torch.gather(input=xy_matrix, index=order_k[:, :self.num_corr, :, :], dim=1)
        select_k_copy = select_k.clone()
        select_k_copy[:,:,:,:6] = torch.flip(select_k[:,:,:,:6].reshape(select_k.shape[0], select_k.shape[1],select_k.shape[2],2,3),dims=(3,)).reshape(select_k.shape[0], select_k.shape[1],select_k.shape[2],6)
        select = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
        select_copy = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
        N_select = select.shape[1]

        pair_matrix = self.pair_head(select.reshape(B * N_select * self.num_corr, 7)).reshape(B * N_select, self.num_corr, self.decoder_embed_dim)
        pair_matrix += get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, select_copy.reshape(B * N_select, self.num_corr, 7)[:,:,3:5])
        x = torch.cat([self.cls_token_2.repeat(B*N_select, 1, 1), pair_matrix], dim=1)
        for blk in self.blocks_2:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.pair_head_2(x[:,0,:].reshape(B*N_select, self.decoder_embed_dim)).reshape(B, N_select, self.decoder_embed_dim)
        x = x.reshape(B, N_select, self.decoder_embed_dim) + get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, select_copy[:,:,0,0:2])
        x = torch.cat([self.cls_token.repeat(B, 1, 1), x], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        if self.num_classes == 1:
            local_score = self.decoder_pred(x[:, 0]).reshape(-1)
            local_score = torch.sigmoid(local_score)
            final_score = global_score.detach() * self.ratio.clamp(min=0.1, max=0.9) + local_score.detach() * (
                    1 - self.ratio.clamp(min=0.1, max=0.9))
        elif self.num_classes == 2:
            local_score = self.decoder_pred(x[:, 0])
            final_score = global_score.detach() * self.ratio.clamp(min=0.1, max=0.9) + self.sm(local_score).detach()[:,1] * (
                    1 - self.ratio.clamp(min=0.1, max=0.9))
        return local_score, final_score


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:,:, 0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:,:, 1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32).cuda()
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    # pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('bm,d->bmd', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=2)  # (M, D)
    return emb


def pos_resize(matrix, size):
    B, C, H, W = matrix.shape
    new_matrix = F.interpolate(matrix, size=size, mode='bicubic')
    ori_dis = ((matrix[:, :, 0, W//2]-matrix[:, :, H//2, W//2])**2).sum(dim=1)
    new_dis = ((new_matrix[:, :, size[0]//2-H//2, size[1] // 2] - new_matrix[:, :, size[0] // 2, size[1] // 2]) ** 2).sum(dim=1)
    ratio = torch.sqrt(ori_dis/new_dis)
    center = new_matrix[:, :, size[0]//2, size[1]//2].unsqueeze(2).unsqueeze(3)
    new_matrix = (new_matrix - center) * ratio + center
    return new_matrix

class res50(torch.nn.Module):
    def __init__(self, num_classes=256, num_feature=8, use_gem=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_feature = num_feature
        self.use_gem = use_gem
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.head = nn.Linear(2048, self.num_classes)
        self.pool = self.resnet.avgpool
        # self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()
        self.l2_norm = L2Norm()
        self.multi_out = False
        if self.use_gem:
            self.pool = GeM(p=3, eps=1e-6)

    def forward(self, x):
        # feature = self.resnet(x).reshape(x.shape[0], 2048, 7, 7)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        feature = self.resnet.layer4(x)
        embed = self.head(torch.flatten(self.pool(feature),start_dim=1))
        return self.l2_norm(embed), feature


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

# To support arbitrary input resolution
class AnySizePatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.parameter.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        # return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'

