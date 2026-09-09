import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append(".")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import iSAID
from model.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss, JointLoss
from model.model_seg_neg_fp5 import network
from torch import autograd
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer

from model.dino_module import DinoFeaturizer
from utils.camutils_g import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg,dino_mask
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
torch.hub.set_dir("./pretrained")

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default='isaid', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='./datasets/iSAID', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=16, type=int, help="number of classes")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_isaid", type=str, help="work_dir_isaid_wseg")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--test_set", default="val", type=str, help="testing split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=2000, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--start_eval_iters", default=10000, type=int, help="iters to start validation")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), help="multi_scales for cam")

parser.add_argument("--w_ptc", default=0.3, type=float, help="w_ptc")
parser.add_argument("--w_ctc", default=0.5, type=float, help="w_ctc")
parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--w_joi", default=0.3, type=float, help="w_joi")
parser.add_argument("--w_entropy", default=0.1, type=float, help="w_entropy")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--bkg_proto_path", default='./background_prototypes/vision_prototypes_isaid.pth', type=str, help="bkg proto_path")
parser.add_argument("--bkg_threshold", default=0.5, type=float, help="sim")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default="1",action="store_true", help="save_ckpt")

parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)

def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_single_class(cls_label):
    pos_counts = torch.sum(cls_label > 0, dim=1)
    
    is_single = torch.zeros_like(pos_counts, dtype=torch.bool)
    class_idx = torch.full_like(pos_counts, -1, dtype=torch.int64)
    
    bg_mask = (pos_counts == 0)
    is_single[bg_mask] = True
    class_idx[bg_mask] = -1
    
    single_mask = (pos_counts == 1)
    is_single[single_mask] = True
    
    if torch.any(single_mask):
        single_samples = cls_label[single_mask]
        single_class_indices = torch.argmax(single_samples, dim=1)
        class_idx[single_mask] = single_class_indices
    
    multi_mask = (pos_counts > 1)
    is_single[multi_mask] = False
    class_idx[multi_mask] = -2
    
    return is_single, class_idx

def create_background_label(refined_pseudo_label, device, ignore_index=255):
    background_label = torch.zeros_like(refined_pseudo_label, dtype=torch.long, device=device)
    background_label = torch.where(
        refined_pseudo_label == ignore_index,
        ignore_index,
        background_label
    )
    return background_label

def compute_similarity_with_prototypes(dino_map, background_prototypes):
    B, C, H, W = dino_map.shape
    K = background_prototypes.shape[0]
    
    dino_flat = dino_map.view(B, C, -1).permute(0, 2, 1)
    
    dino_norm = F.normalize(dino_flat, p=2, dim=-1)
    proto_norm = F.normalize(background_prototypes, p=2, dim=-1)
    
    similarity = torch.matmul(dino_norm, proto_norm.transpose(0, 1))
    
    max_similarity, _ = torch.max(similarity, dim=-1)
    
    max_similarity = max_similarity.view(B, H, W)
    
    return max_similarity

def validate(model=None, data_loader=None, args=None, dino_head=None, background_prototypes=None):
    preds, gts, cams, cams_aux = [], [], [], []
    preds_refined = []
    
    model.eval()
    dino_head.eval()
    
    avg_meter = AverageMeter()
    count = 0
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _, _ = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            
            seg_pred = torch.argmax(resized_segs, dim=1)
            
            dino_map = dino_head(inputs)
            
            bg_similarity = compute_similarity_with_prototypes(dino_map, background_prototypes)
            
            bg_similarity_resized = F.interpolate(
                bg_similarity.unsqueeze(1), 
                size=labels.shape[1:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            
            bg_mask = bg_similarity_resized > args.bkg_threshold
            
            seg_pred_refined = seg_pred.clone()
            seg_pred_refined[bg_mask] = 0
            
            preds += list(seg_pred.cpu().numpy().astype(np.int16))
            preds_refined += list(seg_pred_refined.cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

    cls_score = avg_meter.pop('cls_score')
    
    seg_score = evaluate.scores(gts, preds, args.num_classes)
    cam_score = evaluate.scores(gts, cams, args.num_classes)
    cam_aux_score = evaluate.scores(gts, cams_aux, args.num_classes)
    
    seg_score_refined = evaluate.scores(gts, preds_refined, args.num_classes)
    
    model.train()
    dino_head.train()

    tab_results, mIoU = format_tabs(
        [cam_score, cam_aux_score, seg_score, seg_score_refined], 
        name_list=["CAM", "aux_CAM", "Seg_Pred", "Seg_Refined"],
        cat_list=iSAID.class_list
    )

    return cls_score, tab_results, mIoU

def train(args=None):
    torch.cuda.set_device(args.local_rank)
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))
    mIoU = 0.0001
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = iSAID.iSAIDClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = iSAID.iSAIDSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.test_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_momentum=args.momentum,
        aux_layer=args.aux_layer
    )
    param_groups = model.get_param_groups()
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total:,} ({total/1e6:.2f}M)")

    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 8,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 8,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    dino_head = DinoFeaturizer(70)
    dino_head.to(device)
    
    logging.info(f"加载背景原型: {args.bkg_proto_path}")
    if os.path.exists(args.bkg_proto_path):
        bg_proto_data = torch.load(args.bkg_proto_path, map_location=device)
        background_prototypes = bg_proto_data['prototypes'].to(device)
        logging.info(f"背景原型加载成功，形状: {background_prototypes.shape}")
        logging.info(f"背景原型数量: {bg_proto_data.get('K', 'N/A')}")
    else:
        logging.warning(f"背景原型文件不存在: {args.bkg_proto_path}")
        background_prototypes = torch.randn(3, 256).to(device)
        logging.info(f"使用随机背景原型，形状: {background_prototypes.shape}")

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    ncrops = 10
    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=args.temp).cuda()
    JOINT_loss = JointLoss().cuda()
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in range(args.max_iters):
        try:
            img_name, inputs, cls_label, img_box, crops, = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops, = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        b, c, h, w = inputs.shape
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        is_single_class, class_indices = check_single_class(cls_label)

        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
      
        roi_mask = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)

        local_crops, flags = crop_from_roi_neg(images=crops[2], roi_mask=roi_mask, crop_num=ncrops-2, crop_size=args.local_crop_size)
        roi_crops = crops[:2] + local_crops
        
        cls, segs, foreground_seg, fmap, cls_aux, out_t, out_s = model(inputs, crops=roi_crops, n_iter=n_iter)

        dino_map = dino_head(inputs)

        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        ctc_loss = CTC_loss(out_s, out_t, flags)

        valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box)
        
        background_label = create_background_label(refined_pseudo_label, device, args.ignore_index)
        
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        foreground_seg = F.interpolate(foreground_seg, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        
        mixed_label = refined_pseudo_label.clone()
        for b_idx in range(b):
            if is_single_class[b_idx] and class_indices[b_idx] == -1:
                mixed_label[b_idx] = background_label[b_idx]
        
        seg_loss = get_seg_loss(
            segs, 
            mixed_label.type(torch.long), 
            ignore_index=args.ignore_index
        )
        
        reg_loss = get_energy_loss(
            img=inputs, 
            logit=segs, 
            label=mixed_label, 
            img_box=img_box, 
            loss_layer=loss_layer
        )

        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        
        background_label_aux = create_background_label(pseudo_label_aux, device, args.ignore_index)
        mixed_label_aux = pseudo_label_aux.clone()
        for b_idx in range(b):
            if is_single_class[b_idx] and class_indices[b_idx] == -1:
                mixed_label_aux[b_idx] = background_label_aux[b_idx]
        
        aff_mask = dino_mask(mixed_label_aux, dino_map)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        joint_loss = JOINT_loss(foreground_seg, segs, mixed_label.type(torch.long))

        segs_softmax = F.softmax(segs, dim=1)
        entropy = -torch.sum(segs_softmax * torch.log(segs_softmax + 1e-10), dim=1)
        max_entropy = torch.log(torch.tensor(args.num_classes, dtype=torch.float32, device=device))
        entropy_coeff = entropy / max_entropy

        foreground_seg_resized = F.interpolate(foreground_seg, size=segs.shape[2:], mode='bilinear', align_corners=False)
        foreground_pred = torch.argmax(foreground_seg_resized, dim=1)

        fmap_resized = F.interpolate(fmap, size=segs.shape[2:], mode='bilinear', align_corners=False)
        B, C, H, W = fmap_resized.shape

        pos_mask = (foreground_pred == 0).float()
        neg_mask = (foreground_pred >= 1).float()

        pos_features = []
        neg_features = []

        for b_idx in range(B):
            pos_feat_b = fmap_resized[b_idx]
            pos_mask_b = pos_mask[b_idx].unsqueeze(0)
            if pos_mask_b.sum() > 0:
                weight_b = entropy_coeff[b_idx].unsqueeze(0)
                pos_weighted = weight_b * pos_mask_b
                pos_feat_mean = torch.sum(pos_feat_b * pos_weighted, dim=(1, 2)) / (pos_weighted.sum() + 1e-10)
                pos_features.append(pos_feat_mean)
            
            neg_mask_b = neg_mask[b_idx].unsqueeze(0)
            if neg_mask_b.sum() > 0:
                weight_b = entropy_coeff[b_idx].unsqueeze(0)
                neg_weighted = weight_b * neg_mask_b
                neg_feat_mean = torch.sum(pos_feat_b * neg_weighted, dim=(1, 2)) / (neg_weighted.sum() + 1e-10)
                neg_features.append(neg_feat_mean)

        if len(pos_features) > 0 and len(neg_features) > 0:
            pos_features = torch.stack(pos_features)
            neg_features = torch.stack(neg_features)
            
            pos_sim = F.cosine_similarity(pos_features.unsqueeze(1), pos_features.unsqueeze(0), dim=2)
            neg_sim = F.cosine_similarity(pos_features.unsqueeze(1), neg_features.unsqueeze(0), dim=2)
            
            margin = 0.5
            pos_loss = torch.mean(1.0 - pos_sim)
            neg_loss = torch.mean(torch.clamp(neg_sim + margin, min=0))
            
            batch_entropy_mean = entropy_coeff.mean(dim=(1, 2))
            if len(batch_entropy_mean) >= max(len(pos_features), len(neg_features)):
                entropy_weight_pos = batch_entropy_mean[:len(pos_features)].mean() if len(pos_features) > 0 else 1.0
                entropy_weight_neg = batch_entropy_mean[:len(neg_features)].mean() if len(neg_features) > 0 else 1.0
            else:
                entropy_weight_pos = 1.0
                entropy_weight_neg = 1.0
            
            entropy_contrast_loss = entropy_weight_pos * pos_loss + entropy_weight_neg * neg_loss
        else:
            entropy_contrast_loss = torch.tensor(0.0, device=device)

        if n_iter <= 2000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss + 0.0 * seg_loss + 0.0 * reg_loss + 0.0 * joint_loss + args.w_entropy * entropy_contrast_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_ctc * ctc_loss + args.w_seg * seg_loss + args.w_reg * reg_loss + args.w_joi * joint_loss + args.w_entropy * entropy_contrast_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'ctc_loss': ctc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'joint_loss': joint_loss.item(),
            'entropy_contrast_loss': entropy_contrast_loss.item(),
            'cls_score': cls_score.item(),
        })

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:
            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, ctc_loss: %.4f, seg_loss: %.4f, joint_loss: %.4f, entropy_contrast_loss: %.4f..." % 
                           (n_iter + 1, delta, eta, cur_lr, 
                            avg_meter.pop('cls_loss'), 
                            avg_meter.pop('cls_loss_aux'), 
                            avg_meter.pop('ptc_loss'), 
                            avg_meter.pop('ctc_loss'), 
                            avg_meter.pop('seg_loss'),
                            avg_meter.pop('joint_loss'),
                            avg_meter.pop('entropy_contrast_loss')))

        if (n_iter + 1) >= args.start_eval_iters and (n_iter + 1 - args.start_eval_iters) % args.eval_iters == 0:
            if args.local_rank == 0:
                logging.info('Validating...')
            val_cls_score, tab_results, mIoU_result = validate(
                model=model, 
                data_loader=val_loader, 
                args=args,
                dino_head=dino_head,
                background_prototypes=background_prototypes
            )
            if args.save_ckpt and (n_iter + 1) >= 7000 and mIoU_result[3] > mIoU:
                mIoU = mIoU_result[3]
                ckpt_name = os.path.join(args.ckpt_dir,
                                         "Best mIoU: {}, model: {} model_iter_%d.pth".format(mIoU, "iSAID") % (
                                                 n_iter + 1))
                torch.save(model.state_dict(), ckpt_name)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n" + tab_results)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    setup_seed(args.seed)
    train(args=args)