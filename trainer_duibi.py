import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torch.nn import functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import test_single_volume


def trainer_acdc(args, model, snapshot_path):
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator([args.img_size, args.img_size])]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.00015)
    scheduler = CosineAnnealingLR(optimizer, T_max=3 * args.max_epochs // 4, eta_min=0.000001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("{} val iterations per epoch".format(len(valloader)))

    iter_num = 0
    best_performance = 0.0
    max_epoch = args.max_epochs
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.2 * loss_ce + 0.8 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:  # 500
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    metric_i = test_single_volume(image, label, model, classes=num_classes,
                                                  patch_size=[args.img_size, args.img_size])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_iteration, best_performance, best_hd95 = iter_num, performance, mean_hd95
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info('Best model | iteration %d : mean_dice : %f mean_hd95 : %f' % (
                        iter_num, performance, mean_hd95))

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()
            scheduler.step()
            if iter_num >= max_iterations:
                break


def complementary_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
    normalized_loss = loss / num_pixels
    return normalized_loss


class MultiClassUncertaintyContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_classes=9):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, pred, target, uc_mask_low_res, features):

        target = target.squeeze(1) if target.dim() == 4 else target

        # 上采样不确定掩码
        uc_mask = F.interpolate(uc_mask_low_res.float(),
                                size=target.shape[-2:],
                                mode='bilinear',
                                align_corners=False) > 0.5

        # 多分类概率计算
        pred_prob = torch.softmax(pred, dim=1)
        #
        # # 基于熵的权重计算
        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-10), dim=1)
        difficulty_weights = entropy / torch.log(torch.tensor(self.num_classes,
                                                              device=entropy.device))

        # 样本选择
        masked_weights = difficulty_weights * uc_mask.squeeze(1).float()
        low_conf = torch.zeros_like(uc_mask, dtype=torch.bool)

        for b in range(pred.shape[0]):
            flat_weights = masked_weights[b].flatten()
            flat_uc = uc_mask[b].flatten()
            valid_weights = flat_weights[flat_uc]
            if len(valid_weights) > 0:
                k = max(1, int(len(valid_weights) * 0.6))
                threshold = torch.topk(valid_weights, k=k).values[-1]
                low_conf[b] = (masked_weights[b] >= threshold) & uc_mask[b]
        # low_conf = uc_mask
        pixel_losses = []
        for b in range(pred.shape[0]):
            uc_y, uc_x = torch.where(low_conf[b, 0])
            for y, x in zip(uc_y, uc_x):
                loss = self._calc_pixel_loss(
                    features[b], target[b], pred_prob[b], y, x
                )
                if loss is not None:
                    pixel_losses.append(loss)

        return torch.mean(torch.stack(pixel_losses)) if pixel_losses else 0.0

    def _calc_pixel_loss(self, features, target, pred_prob, y, x):
        # print(features.shape,target.shape,pred_prob.shape)
        H, W = target.shape
        y = max(1, min(H - 2, y))
        x = max(1, min(W - 2, x))

        # 获取邻域特征和标签
        neighbor_feats = features[:, y - 1:y + 2, x - 1:x + 2].flatten(1)  # [C,9]
        neighbor_gt = target[y - 1:y + 2, x - 1:x + 2].flatten()  # [9]

        # 多分类正负样本定义
        pos_mask = (neighbor_gt == target[y, x])
        mask = torch.ones(9, dtype=bool, device=features.device)
        mask[4] = False  # 排除中心点

        pos_feats = neighbor_feats[:, mask & pos_mask].t()  # [N_pos,C]
        neg_feats = neighbor_feats[:, mask & ~pos_mask].t()  # [N_neg,C]

        if len(pos_feats) == 0 or len(neg_feats) == 0:
            return None

        # 对比损失计算
        anchor = features[:, y, x].unsqueeze(0)
        pos_sim = F.cosine_similarity(anchor, pos_feats) / self.temperature
        neg_sim = F.cosine_similarity(anchor, neg_feats) / self.temperature

        logits = torch.cat([pos_sim, neg_sim])
        labels = torch.cat([torch.ones_like(pos_sim),
                            torch.zeros_like(neg_sim)])

        return F.binary_cross_entropy_with_logits(logits, labels)


from trainer_SCCL_BASE import *


def trainer_synapse(args, model, snapshot_path, contrasive_fn=MultiClassUncertaintyContrastiveLoss(),
                    contrasive_fn1=SCCL1(temperature=0.1, num_positives=10, num_negatives=10, sample_ratio=0.25,
                                         ignore_class=0)):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs, out2, out1, fg_pred, uc_pred, d3, d2, au2, au1 = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss1 = 0.5 * loss_ce + 0.5 * loss_dice
            loss2 = 0.5 * ce_loss(out1, label_batch[:].long()) + 0.5 * dice_loss(out1, label_batch, softmax=True)
            loss3 = 0.5 * ce_loss(out2, label_batch[:].long()) + 0.5 * dice_loss(out2, label_batch, softmax=True)
            loss_fg = 0.5 * ce_loss(fg_pred, label_batch[:].long()) + 0.5 * dice_loss(fg_pred, label_batch,
                                                                                      softmax=True)

            loss_mask = loss1 + 0.4 * loss2 + 0.4 * loss3

            foreground_prob = torch.max(fg_pred[:, 1:9, :, :], dim=1)[0]  # 形状 [B, H, W]
            # 2. 背景概率（直接取第0类）
            background_prob = fg_pred[:, 0, :, :]  # 形状 [B, H, W]
            # 3. 拼接背景和前景概率
            combined = torch.stack([background_prob, foreground_prob], dim=1)  # 形状 [B, 2, H, W]
            # 4. Softmax归一化
            final_probs = F.softmax(combined, dim=1)  # 形状 [B, 2, H, W]
            preds = torch.stack(
                [final_probs[:, 1, :, :].unsqueeze(1), final_probs[:, 0, :, :].unsqueeze(1), torch.sigmoid(uc_pred)],
                dim=1)
            probs = F.softmax(preds, dim=1)

            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]
            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
            loss_comp = loss_comp.cuda()

            pred_labels = torch.argmax(probs, dim=1)  # shape: [B, H, W]
            uc_mask = (pred_labels == 2)
            y11 = F.interpolate(label_batch[:].unsqueeze(1).float(), scale_factor=1 / 8, mode='nearest')

            y12 = F.interpolate(label_batch[:].unsqueeze(1).float(), scale_factor=1 / 4, mode='nearest')

            contrasive_loss1 = contrasive_fn(au1, y11, uc_mask, d3)
            contrasive_loss2 = contrasive_fn(au2, y12,uc_mask, d2)

            contrasive_loss = contrasive_loss1 + contrasive_loss2

            ###############################################################################
            y11 = F.interpolate(label_batch[:].unsqueeze(1).float(), scale_factor=1 / 8, mode='nearest')
            y12 = F.interpolate(label_batch[:].unsqueeze(1).float(), scale_factor=1 / 4, mode='nearest')

            # print(y11.shape,au1.shape)
            contrasive_loss1_ = contrasive_fn1(d3, y11, au1)
            contrasive_loss2_ = contrasive_fn1(d2, y12, au2)

            contrasive_loss_ = contrasive_loss1_ + contrasive_loss2_
            loss = loss_mask + loss_fg + loss_comp + 0.1 * contrasive_loss + 0.1 * contrasive_loss_
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        scheduler.step()
        save_interval = 50  # int(max_epoch/6)
        save_interval = 1
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
