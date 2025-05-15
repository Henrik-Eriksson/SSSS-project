import argparse
from copy import deepcopy
import logging
import os
import pprint
import yaml
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

# ----------------------------
# LogicDiag imports
# ----------------------------
from logicdiag import (
    CompositionRule,
    DecompositionRule,
    ExclusionRule,
    resolve_conflicts,
)

# -----------------------------------------------------------------------------
# Helper to load the first‑order rules from a YAML file.
# -----------------------------------------------------------------------------

def load_rules(yaml_path):
    """Return a list[Rule] understood by LogicDiag.

    Expected YAML structure::

        composition:
          - parent: 0
            children: [1, 2]
        decomposition:
          - parent: 1
            children: [3, 4]
        exclusion:
          - group: [2, 5, 6]
    """
    with open(yaml_path, "r") as f:
        spec = yaml.safe_load(f)

    rules = []
    for item in spec.get("composition", []):
        rules.append(CompositionRule(item["parent"], item["children"]))
    for item in spec.get("decomposition", []):
        rules.append(DecompositionRule(item["parent"], item["children"]))
    for item in spec.get("exclusion", []):
        rules.append(ExclusionRule(item["group"]))
    return rules

# -----------------------------------------------------------------------------
# Convert per‑pixel logits into LogicDiag‑consistent hard labels (pseudo labels).
# -----------------------------------------------------------------------------

def logicdiag_labels(logits, K, max_diag_size=2):
    """Apply LogicDiag to each pixel separately.

    Args:
        logits: [B, C, H, W] tensor *before* softmax.
        K: list of rules.
    Returns:
        mask: LongTensor [B, H, W] with the chosen class index per pixel.
    """
    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N, C]
    with torch.no_grad():
        O_fixed = resolve_conflicts(logits_flat, K, max_diag_size=max_diag_size)
    mask = O_fixed.argmax(dim=1).view(B, H, W).long()
    return mask

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="UniMatch‑LogicDiag: Threshold‑free semi‑supervised semantic segmentation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--labeled-id-path", type=str, required=True)
parser.add_argument("--unlabeled-id-path", type=str, required=True)
parser.add_argument("--rules-path", type=str, required=True, help="YAML file with LogicDiag rules")
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    K = load_rules(args.rules_path)

    logger = init_log("global", logging.INFO)
    logger.propagate = False

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # ------------------------------------------------------------------
    # Model construction (unchanged from original UniMatch V2)
    # ------------------------------------------------------------------
    model_configs = {
        "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
        "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
        "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = DPT(**{**model_configs[cfg["backbone"].split("_")[-1]], "nclass": cfg["nclass"]})
    state_dict = torch.load(f"./pretrained/{cfg['backbone']}.pth", map_location="cpu")
    model.backbone.load_state_dict(state_dict)

    if cfg.get("lock_backbone", False):
        model.lock_backbone()

    optimizer = AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg["lr"]},
            {
                "params": [param for name, param in model.named_parameters() if "backbone" not in name],
                "lr": cfg["lr"] * cfg["lr_multi"],
            },
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    if rank == 0:
        logger.info("Total params: {:.1f}M".format(count_params(model)))
        logger.info("Encoder params: {:.1f}M".format(count_params(model.backbone)))
        logger.info("Decoder params: {:.1f}M\n".format(count_params(model.head)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )

    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Losses (same as original)
    # ------------------------------------------------------------------
    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError("%s criterion is not implemented" % cfg["criterion"]["name"])

    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)

    # ------------------------------------------------------------------
    # Datasets & loaders (unchanged)
    # ------------------------------------------------------------------
    trainset_u = SemiDataset(cfg["dataset"], cfg["data_root"], "train_u", cfg["crop_size"], args.unlabeled_id_path)
    trainset_l = SemiDataset(
        cfg["dataset"], cfg["data_root"], "train_l", cfg["crop_size"], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(cfg["dataset"], cfg["data_root"], "val")

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_l,
    )

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler_u,
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    last_epoch = -1

    # Optionally resume
    ckpt_path = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        last_epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        previous_best_ema = checkpoint["previous_best_ema"]
        best_epoch = checkpoint["best_epoch"]
        best_epoch_ema = checkpoint["best_epoch_ema"]
        if rank == 0:
            logger.info("************ Resumed from epoch %i\n" % last_epoch)

    for epoch in range(last_epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, EMA: {:.2f} @epoch-{:}".format(
                    epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema
                )
            )

        # meters
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()

        # epoch‑wise shuffling
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        model.train()

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
        ) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()

            # ------------------------------------------------------------------
            # 1) Generate LogicDiag‑consistent pseudo labels from EMA teacher
            # ------------------------------------------------------------------
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()  # [B, C, H, W]
                mask_u_w = logicdiag_labels(pred_u_w, K)  # [B, H, W]

            # ------------------------------------------------------------------
            # 2) CutMix augmentation (unchanged)
            # ------------------------------------------------------------------
            img_u_s1[cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1] = img_u_s1.flip(0)[
                cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1
            ]
            img_u_s2[cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1] = img_u_s2.flip(0)[
                cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1
            ]

            # ------------------------------------------------------------------
            # 3) Student forward + supervised & unsupervised losses
            # ------------------------------------------------------------------
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)

            # create CutMix‑aware pseudo labels
            mask_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]

            # ----- Supervised loss
            loss_x = criterion_l(pred_x, mask_x)

            # ----- Unsupervised loss (no confidence threshold)
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * (ignore_mask_cutmixed1 != 255)
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * (ignore_mask_cutmixed2 != 255)
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_s = 0.5 * (loss_u_s1 + loss_u_s2)
            loss = 0.5 * (loss_x + loss_u_s)

            # ------------------------------------------------------------------
            # 4) Optimisation & EMA update
            # ------------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())

            iters = epoch * len(trainloader_u) + i
            lr_factor = (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = cfg["lr"] * lr_factor
            optimizer.param_groups[1]["lr"] = cfg["lr"] * cfg["lr_multi"] * lr_factor

            # EMA
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.data.mul_(ema_ratio).add_(param.data, alpha=1 - ema_ratio)
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.data.mul_(ema_ratio).add_(buffer.data, alpha=1 - ema_ratio)

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s.item(), iters)

                if i % (len(trainloader_u) // 8) == 0:
                    logger.info(
                        "Iters: {:}, LR: {:.7f}, Total: {:.3f}, Lx: {:.3f}, Lu: {:.3f}".format(
                            i, optimizer.param_groups[0]["lr"], total_loss.avg, total_loss_x.avg, total_loss_s.avg
                        )
                    )

        # ------------------------------------------------------------------
        # Validation (unchanged)
        # ------------------------------------------------------------------
        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** Class [{:} {:}] IoU: {:.2f}, EMA: {:.2f}".format(
                        cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou, iou_class_ema[cls_idx]
                    )
                )
            logger.info(
                "***** Evaluation {} ***** >>>> mIoU: {:.2f}, EMA: {:.2f}\n".format(eval_mode, mIoU, mIoU_ema)
            )
            writer.add_scalar("eval/mIoU", mIoU, epoch)
            writer.add_scalar("eval/mIoU_ema", mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar(f"eval/{CLASSES[cfg['dataset']][i]}_IoU", iou, epoch)
                writer.add_scalar(f"eval/{CLASSES[cfg['dataset']][i]}_IoU_ema", iou_class_ema[i], epoch)

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if mIoU >= previous_best:
            best_epoch = epoch
        if mIoU_ema >= previous_best_ema:
            best_epoch_ema = epoch
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)

        if rank == 0:
            ckpt = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "previous_best_ema": previous_best_ema,
                "best_epoch": best_epoch,
                "best_epoch_ema": best_epoch_ema,
            }
            torch.save(ckpt, os.path.join(args.save_path, "latest.pth"))
            if mIoU >= previous_best:
                torch.save(ckpt, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    main()
