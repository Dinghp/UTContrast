import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

# =========================
# 1. 你提供的：边缘检测函数（vectorized）
# =========================
def edge_contour_vectorized(label, ignore_index=255, background_index=0):
    original_device = label.device
    label = label.squeeze(0)
    label_np = label.cpu().numpy().astype(np.int32)
    #  print(label_np.shape)
    b, h, w = label_np.shape

    edge = np.zeros((b, h, w), dtype=np.uint8)

    for i in range(b):
        current_label = label_np[i]
        valid_mask = (current_label != ignore_index)

        right_diff = np.zeros((h, w), dtype=bool)
        right_diff[:, :-1] = (current_label[:, :-1] != current_label[:, 1:]) & \
                             (current_label[:, :-1] != ignore_index) & \
                             (current_label[:, 1:] != ignore_index)

        down_diff = np.zeros((h, w), dtype=bool)
        down_diff[:-1, :] = (current_label[:-1, :] != current_label[1:, :]) & \
                            (current_label[:-1, :] != ignore_index) & \
                            (current_label[1:, :] != ignore_index)

        combined_edges = right_diff | down_diff
        combined_edges[current_label == background_index] = False
        edge[i] = combined_edges.astype(np.uint8)

    edge_tensor = torch.from_numpy(edge).float().to(original_device)
    return edge_tensor


# =========================
# 2. 辅助函数：获取预测错误的像素索引（线性索引）
# =========================
def get_misclassified_indices(pred_labels, class_labels):
    #  print(pred_labels.shape)
    assert pred_labels.shape == class_labels.shape
    correct_mask = (pred_labels == class_labels)
    mis_mask = ~correct_mask
    mis_idx_list = []
    B, H, W = class_labels.shape
    for b in range(B):
        mis_idx = (mis_mask[b]).nonzero()  # [N_mis, 2] -> y,x
        y_coords = mis_idx[:, 0]
        x_coords = mis_idx[:, 1]
        linear_idx = y_coords * W + x_coords  # flatten index
        mis_idx_list.append(linear_idx)  # List[Tensor], 每个是该图上错误像素的 linear index
    return mis_idx_list


# =========================
# 3. 辅助函数：计算预测错误像素到边缘的最小距离，选 top 20%
# =========================
# def compute_misclassified_to_edge_distance(mis_idx_list, class_labels, spatial_shapes):
#     B = len(mis_idx_list)
#     edge_maps = edge_contour_vectorized(class_labels)  # [B, H, W]
#     all_selected_indices = []
#
#     for b in range(B):
#         mis_linear_idx = mis_idx_list[b]  # [N_mis, ]
#         if len(mis_linear_idx) == 0:
#             all_selected_indices.append(torch.empty((0,), device=mis_linear_idx.device, dtype=torch.long))
#             continue
#
#         H, W = spatial_shapes[b]
#         y_coords = mis_linear_idx // W
#         x_coords = mis_linear_idx % W
#         mis_coords = torch.stack([y_coords, x_coords], dim=1)  # [N_mis, 2]
#
#         edge_map = edge_maps[b]  # [H, W]
#         edge_coords = (edge_map == 1).nonzero()  # [N_edge, 2], format: [y, x]
#
#         if len(edge_coords) == 0:
#             all_selected_indices.append(mis_linear_idx)
#             continue
#
#         edge_y = edge_coords[:, 0].view(1, -1)
#         edge_x = edge_coords[:, 1].view(1, -1)
#         mis_y = y_coords.view(-1, 1)
#         mis_x = x_coords.view(-1, 1)
#
#         dists = torch.sqrt((mis_y - edge_y).pow(2) + (mis_x - edge_x).pow(2))  # [N_mis, N_edge]
#         min_dists, _ = torch.min(dists, dim=1)  # [N_mis,]
#
#         num_mis = len(min_dists)
#         num_select = max(1, int(num_mis * 0.2))
#         _, sorted_idx = torch.topk(min_dists, k=num_select, largest=False)
#         selected_mis_linear = mis_linear_idx[sorted_idx]
#
#         all_selected_indices.append(selected_mis_linear)
#
#     return all_selected_indices
#
#
# # =========================
# # 4. 主模块：SCCL（多类别对比学习 + 高质量负样本）
# # =========================
# class SCCL(nn.Module):
#     def __init__(self, temperature=0.1, num_positives=4, num_negatives=10, sample_ratio=0.5, ignore_class=0):
#         super().__init__()
#         self.temperature = temperature
#         self.num_positives = num_positives
#         self.num_negatives = num_negatives
#         self.sample_ratio = sample_ratio
#         self.ignore_class = ignore_class
#         self.cross_entropy = nn.CrossEntropyLoss()
#
#     def forward(self, feat, class_labels, pred_logits=None):
#         B, C, H, W = feat.shape
#         device = feat.device
#
#         feat_norm = F.normalize(feat, p=2, dim=1)  # [B, C, H, W]
#         total_loss = 0.0
#         total_pairs = 0
#
#         spatial_shapes = [(class_labels[b].shape[0], class_labels[b].shape[1]) for b in range(B)]
#
#         for b in range(B):
#             feat_b = feat_norm[b]  # [C, H, W]
#             feat_flat = feat_b.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
#             labels_b = class_labels[b]  # [H, W]
#             labels_flat = labels_b.view(-1)  # [H*W]
#
#             valid_indices = (labels_flat != self.ignore_class).nonzero().squeeze(1)
#             if len(valid_indices) < 2:
#                 continue
#
#             num_valid = len(valid_indices)
#             num_queries = max(1, int(num_valid * self.sample_ratio))
#             query_indices = valid_indices[torch.randperm(num_valid, device=device)[:num_queries]]  # [Q, ]
#
#             # ==========================
#             # 【新增】预测错误像素筛选流程
#             # ==========================
#             selected_mis_indices = None
#             if pred_logits is not None:
#                 pred_logit_b = pred_logits[b]  # [9, H, W]
#                 pred_prob_b = F.softmax(pred_logit_b, dim=0)
#                 pred_label_b = torch.argmax(pred_prob_b, dim=0)  # [H, W]
#
#                 mis_idx_list = get_misclassified_indices(pred_label_b, labels_b)  # List[Tensor]
#                 mis_idx_b = mis_idx_list[0] if len(mis_idx_list) > 0 else torch.empty((0,), device=device, dtype=torch.long)
#
#                 selected_mis_indices = compute_misclassified_to_edge_distance(
#                     [mis_idx_b], class_labels[b:b+1], spatial_shapes[b:b+1]
#                 )[0]  # [N_top20, ]
#
#             for idx in query_indices:
#                 anchor_label = labels_flat[idx]
#                 if anchor_label == self.ignore_class:
#                     continue
#
#                 anchor_feat = feat_flat[idx]
#
#                 # --- 原始负样本逻辑：不同类别 ---
#                 different_class_indices = (labels_flat != anchor_label).nonzero().squeeze(1)
#                 different_class_indices = different_class_indices[different_class_indices != idx]
#                 negatives = feat_flat[different_class_indices]
#                 sampled_negatives = negatives[torch.randperm(len(negatives), device=device)[:min(len(negatives), self.num_negatives)]]
#
#                 # --- 【新增】高质量负样本：预测错 + 靠近边界 + 不同类 ---
#                 sampled_high_quality_negs = torch.empty((0, C), device=device)
#                 if selected_mis_indices is not None and len(selected_mis_indices) > 0:
#                     mis_feats = feat_flat[selected_mis_indices]
#                     mis_labels = labels_flat[selected_mis_indices]
#                     diff_class_mask = (mis_labels != anchor_label)
#                     diff_class_mis_indices = selected_mis_indices[diff_class_mask]
#                     diff_class_mis_feats = feat_flat[diff_class_mis_indices]
#
#                     if len(diff_class_mis_feats) > 0:
#                         sampled_high_quality_negs = diff_class_mis_feats[
#                             torch.randperm(len(diff_class_mis_feats), device=device)[:min(len(diff_class_mis_feats), 10)]
#                         ]
#
#                 # --- 合并负样本：原始 10个 + 高质量 10个 ---
#                 all_negatives = torch.cat([sampled_negatives, sampled_high_quality_negs], dim=0)  # [N_neg_total, C]
#
#                 # --- 对比损失计算 ---
#                 logits = torch.matmul(anchor_feat.unsqueeze(0), all_negatives.T) / self.temperature  # [1, N_neg]
#                 labels = torch.zeros(1, dtype=torch.long, device=device)
#                 total_loss += self.cross_entropy(logits, labels)
#                 total_pairs += 1
#
#         return total_loss / max(total_pairs, 1) if total_pairs > 0 else torch.tensor(0.0, device=device)
# =========================
# 3. 辅助函数：计算预测错误像素到边缘的最小距离，选 前 5 个（固定数量）
# =========================
def compute_misclassified_to_edge_distance(mis_idx_list, class_labels, spatial_shapes):
    B = len(mis_idx_list)
    edge_maps = edge_contour_vectorized(class_labels)  # [B, H, W]
    all_selected_indices = []

    for b in range(B):
        mis_linear_idx = mis_idx_list[b]  # [N_mis, ]
        if len(mis_linear_idx) == 0:
            all_selected_indices.append(torch.empty((0,), device=mis_linear_idx.device, dtype=torch.long))
            continue

        H, W = spatial_shapes[b]
        y_coords = mis_linear_idx // W
        x_coords = mis_linear_idx % W
        mis_coords = torch.stack([y_coords, x_coords], dim=1)  # [N_mis, 2]

        edge_map = edge_maps[b]  # [H, W]
        edge_coords = (edge_map == 1).nonzero()  # [N_edge, 2], format: [y, x]

        if len(edge_coords) == 0:
            all_selected_indices.append(mis_linear_idx)
            continue

        edge_y = edge_coords[:, 0].view(1, -1)
        edge_x = edge_coords[:, 1].view(1, -1)
        mis_y = y_coords.view(-1, 1)
        mis_x = x_coords.view(-1, 1)

        dists = torch.sqrt((mis_y - edge_y).pow(2) + (mis_x - edge_x).pow(2))  # [N_mis, N_edge]
        min_dists, _ = torch.min(dists, dim=1)  # [N_mis,]

        num_mis = len(min_dists)
        #num_select = min(5, num_mis)  # ✅ 你修改的：固定选 5 个，不是 20%
        num_select = min(10, num_mis)  # ✅ 你修改的：固定选 5 个，不是 20%
        _, sorted_idx = torch.topk(min_dists, k=num_select, largest=False)  # 最小距离在前
        selected_mis_linear = mis_linear_idx[sorted_idx]  # [5, ]

        all_selected_indices.append(selected_mis_linear)

    return all_selected_indices


# =========================
# 4. 主模块：SCCL（多类别对比学习 + 高质量负样本：前5个预测错误像素）
# =========================
class SCCL1(nn.Module):
    def __init__(self, temperature=0.1, num_positives=10, num_negatives=10, sample_ratio=0.5, ignore_class=0):
        super().__init__()
        self.temperature = temperature
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.sample_ratio = sample_ratio
        self.ignore_class = ignore_class
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat, class_labels, pred_logits=None):
        B, C, H, W = feat.shape
        device = feat.device

        feat_norm = F.normalize(feat, p=2, dim=1)  # [B, C, H, W]
        total_loss = 0.0
        total_pairs = 0

        spatial_shapes = [(class_labels[b].shape[0], class_labels[b].shape[1]) for b in range(B)]

        for b in range(B):
            feat_b = feat_norm[b]  # [C, H, W]
            feat_flat = feat_b.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
            labels_b = class_labels[b]  # [H, W]
            labels_flat = labels_b.view(-1)  # [H*W]

            valid_indices = (labels_flat != self.ignore_class).nonzero().squeeze(1)
            if len(valid_indices) < 2:
                continue

            num_valid = len(valid_indices)
            num_queries = max(1, int(num_valid * self.sample_ratio))
            query_indices = valid_indices[torch.randperm(num_valid, device=device)[:num_queries]]  # [Q, ]

            # ==========================
            # 【新增】预测错误像素筛选流程
            # ==========================
            selected_mis_indices = None
            if pred_logits is not None:
                pred_logit_b = pred_logits[b]  # [9, H, W]
                pred_prob_b = F.softmax(pred_logit_b, dim=0)
                pred_label_b = torch.argmax(pred_prob_b, dim=0)  # [H, W]
                # print(pred_label_b.shape,labels_b.shape)
                pred_label_b = pred_label_b.unsqueeze(0)

                mis_idx_list = get_misclassified_indices(pred_label_b, labels_b)  # List[Tensor]
                mis_idx_b = mis_idx_list[0] if len(mis_idx_list) > 0 else torch.empty((0,), device=device,
                                                                                      dtype=torch.long)

                selected_mis_indices = compute_misclassified_to_edge_distance(
                    [mis_idx_b], class_labels[b:b + 1], spatial_shapes[b:b + 1]
                )[0]  # [5, ]

            for idx in query_indices:
                anchor_label = labels_flat[idx]
                if anchor_label == self.ignore_class:
                    continue

                anchor_feat = feat_flat[idx]

                # --- 正样本：同类别，随机选 5 个 ---
                same_class_indices = (labels_flat == anchor_label).nonzero().squeeze(1)
                same_class_indices = same_class_indices[same_class_indices != idx]  # 去掉自己
                if len(same_class_indices) == 0:
                    continue
                positives = feat_flat[same_class_indices]
                sampled_positives = positives[
                    torch.randperm(len(positives), device=device)[:min(len(positives), self.num_positives)]  # ✅ 正样本选5个
                ]

                # --- 普通负样本：不同类别，随机选 10 个 ---
                different_class_indices = (labels_flat != anchor_label).nonzero().squeeze(1)
                # different_class_indices = (
                #             (labels_flat != anchor_label) & (labels_flat != self.ignore_class)).nonzero().squeeze(1)
                different_class_indices = different_class_indices[different_class_indices != idx]
                negatives = feat_flat[different_class_indices]
                sampled_negatives = negatives[
                    torch.randperm(len(negatives), device=device)[:min(len(negatives), self.num_negatives)]
                ]

                # --- 【新增】高质量负样本：预测错 + 靠近边缘（前5） + 不同类 ---
                sampled_high_quality_negs = torch.empty((0, C), device=device)
                if selected_mis_indices is not None and len(selected_mis_indices) > 0:
                    mis_feats = feat_flat[selected_mis_indices]
                    mis_labels = labels_flat[selected_mis_indices]
                    diff_class_mask = (mis_labels != anchor_label)
                    diff_class_mis_indices = selected_mis_indices[diff_class_mask]
                    diff_class_mis_feats = feat_flat[diff_class_mis_indices]

                    if len(diff_class_mis_feats) > 0:
                        sampled_high_quality_negs = diff_class_mis_feats[
                            torch.randperm(len(diff_class_mis_feats), device=device)[:min(len(diff_class_mis_feats), 4)]
                        ]

                all_samples = torch.cat([sampled_positives, sampled_negatives, sampled_high_quality_negs],
                                        dim=0)  # [1+15, C]

                # InfoNCE: 1个正类索引(0) + 其余都是负类
                logits = torch.matmul(anchor_feat.unsqueeze(0), all_samples.T) / self.temperature  # [1, 1+15]
                labels = torch.zeros(1, dtype=torch.long, device=device)  # 指向第0位正样本

                total_loss += self.cross_entropy(logits, labels)
                total_pairs += 1

        return total_loss / max(total_pairs, 1) if total_pairs > 0 else torch.tensor(0.0, device=device)

