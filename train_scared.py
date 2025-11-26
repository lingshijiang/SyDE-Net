import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
# from util import InputPadder
from core.utils.utils import InputPadder
from core.SyDENet import SyDENet
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets_scared as datasets
from core.stereo_datasets_scared import SCAREDListDataset
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pathlib import Path
import os
wandb.init(project="my-awesome-project", mode="offline")

def disparity_epe(prediction, reference, max_disparity=None):
    """计算视差EPE误差，与train_disparity.py保持一致"""
    assert prediction.size() == reference.size()
    assert len(prediction.size()) == 3
    # find all valid gt values
    
    if max_disparity:
        valid_mask = (reference > 0) & (reference < max_disparity)
    else:
        valid_mask = reference > 0
        
    diff = torch.abs(prediction - reference)
    diff[~valid_mask] = 0
    valid_pixels = torch.sum(valid_mask)
    err = torch.sum(diff)
    batch_error = err / valid_pixels
    # reject samples with no disparity values out of range.
    return torch.mean(batch_error[valid_pixels > 0]).detach()


def gray_2_colormap_np(img, cmap='rainbow', max=None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img < 0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    # 使用与disparity_epe一致的有效像素掩码定义
    # 结合原始valid掩码和视差范围
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # 初始预测损失
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    
    # 序列预测损失
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    # 使用统一的disparity_epe函数计算EPE
    # 确保输入是3D张量 (B, H, W)
    disp_pred_final = disp_preds[-1]
    if len(disp_pred_final.shape) == 4:  # (B, 1, H, W)
        disp_pred_final = disp_pred_final.squeeze(1)  # (B, H, W)
    if len(disp_gt.shape) == 4:  # (B, 1, H, W)
        disp_gt_3d = disp_gt.squeeze(1)  # (B, H, W)
    else:
        disp_gt_3d = disp_gt
    
    epe = disparity_epe(disp_pred_final, disp_gt_3d, max_disparity=max_disp)

    # 计算MAE (Mean Absolute Error)
    mae = torch.abs(disp_preds[-1] - disp_gt)
    mae = mae[valid.bool() & ~torch.isnan(mae)].mean()
    
    # 计算D1, bad1, bad2, bad3指标
    disp_pred_final = disp_preds[-1]
    abs_error = torch.abs(disp_pred_final - disp_gt)
    
    # D1: 误差 > 1px 且 误差 > 5% 的像素比例
    d1_error = (abs_error > 1.0) & (abs_error > 0.05 * torch.abs(disp_gt))
    d1_rate = d1_error[valid.bool() & ~torch.isnan(d1_error)].float().mean()
    
    # bad1: 误差 > 1px 的像素比例
    bad1_error = abs_error > 1.0
    bad1_rate = bad1_error[valid.bool() & ~torch.isnan(bad1_error)].float().mean()
    
    # bad2: 误差 > 2px 的像素比例
    bad2_error = abs_error > 2.0
    bad2_rate = bad2_error[valid.bool() & ~torch.isnan(bad2_error)].float().mean()
    
    # bad3: 误差 > 3px 的像素比例
    bad3_error = abs_error > 3.0
    bad3_rate = bad3_error[valid.bool() & ~torch.isnan(bad3_error)].float().mean()
    
    metrics = {
        'train/i_loss_left': i_loss[valid.bool() & ~torch.isnan(i_loss)].mean(),
        'train/epe': epe.mean(),
        'train/mae': mae,
        'train/d1': d1_rate,
        'train/bad1': bad1_rate,
        'train/bad2': bad2_rate,
        'train/bad3': bad3_rate,
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }
    return disp_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters()))
    rest_params = filter(lambda x: id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr / 2.0},
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr / 2.0, args.lr], args.total_step + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


@hydra.main(version_base=None, config_path='config', config_name='train_scared')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16',
                              dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb',
                              kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True),
                              init_kwargs={'wandb': cfg.wandb})

    # 根据配置选择数据集来源：
    # 1) 若提供 list 文件，则使用基于列表的 SCAREDListDataset（参照另一个项目的预处理方式）
    # 2) 否则回退到现有的 fetch_dataloader(cfg)

    use_list_ds = hasattr(cfg, 'data_path') and hasattr(cfg, 'trainlist') and hasattr(cfg, 'testlist') and \
                  os.path.exists(cfg.data_path) and os.path.exists(cfg.trainlist) and os.path.exists(cfg.testlist)

    if use_list_ds:
        print("Loading SCAREDListDataset (list-file based preprocessing)...")
        train_dataset = SCAREDListDataset(aug_params=None, datapath=cfg.data_path, list_filename=cfg.trainlist, training=True)
        val_dataset = SCAREDListDataset(aug_params=None, datapath=cfg.data_path, list_filename=cfg.testlist, training=False)

        print(f"SCAREDListDataset loaded: train={len(train_dataset)}, val={len(val_dataset)}")

        if cfg.total_step == 1:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.batch_size // cfg.num_gpu,
                pin_memory=True,
                shuffle=False,
                num_workers=int(4),
                drop_last=True,
                sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=1)
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.batch_size // cfg.num_gpu,
                pin_memory=True,
                shuffle=True,
                num_workers=int(4),
                drop_last=True
            )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=int(1),
            pin_memory=True,
            shuffle=False,
            num_workers=int(4),
            drop_last=False
        )
    else:
        # 使用SCARED数据集（按现有的 fetch_dataloader 路径，含80/20分割）
        print("Loading SCARED training dataset via fetch_dataloader(cfg)...")
        train_dataset = datasets.fetch_dataloader(cfg)
        print(f"SCARED training dataset loaded: {len(train_dataset)} samples")

        if cfg.total_step == 1:  # 如果只训练一步
            # 只加载一个批次
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size // cfg.num_gpu,
                                                       pin_memory=True, shuffle=False, num_workers=int(4), drop_last=True,
                                                       sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=1))
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size // cfg.num_gpu,
                                                       pin_memory=True, shuffle=True, num_workers=int(4), drop_last=True)

        # 验证集：使用SCARED数据集的验证分割
        print("Loading validation dataset...")
        if hasattr(cfg, 'val_dataset') and cfg.val_dataset == 'scared':
            # 使用SCARED验证集分割（20%的数据）
            val_dataset = datasets.SCARED(aug_params=None, split='val')  # 不使用数据增强
            print("Using SCARED validation dataset (20% split)")
        else:
            # 默认使用SceneFlow测试集作为验证
            val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
            print("Using SceneFlow test dataset for validation")
    
    max_val_samples = getattr(cfg, 'val_samples', len(val_dataset))
    # 如果指定了较小的样本数量，创建一个子集
    if max_val_samples < len(val_dataset):
        # 随机选择指定数量的样本索引
        indices = torch.randperm(len(val_dataset))[:max_val_samples].tolist()
        # 使用PyTorch的Subset类创建子集
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        logging.info(f"Using validation subset with {len(val_dataset)} samples")

    # 创建验证数据加载器
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(1),
                                             pin_memory=True, shuffle=False, num_workers=int(4), drop_last=False)

    # 创建模型
    model = SyDENet(cfg)
    if not cfg.restore_ckpt.endswith("None"):
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]

        # 过滤掉不匹配的键（如注意力模块）
        model_keys = set(model.state_dict().keys())
        ckpt_filtered = {k: v for k, v in ckpt.items() if k in model_keys}
        missing_keys = model_keys - set(ckpt_filtered.keys())
        unexpected_keys = set(ckpt.keys()) - model_keys
        
        if missing_keys:
            print(f"Missing keys in checkpoint: {list(missing_keys)[:10]}...")  # 只显示前10个
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {list(unexpected_keys)[:10]}...")  # 只显示前10个
            
        model.load_state_dict(ckpt_filtered, strict=False)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint
    
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer,
                                                                                   lr_scheduler, val_loader)
    model.to(accelerator.device)

    total_step = 0
    should_keep_training = True
    
    print("Starting training on SCARED dataset...")
    print(f"DataLoader信息:")
    print(f"  - 训练DataLoader长度: {len(train_loader)}")
    print(f"  - 验证DataLoader长度: {len(val_loader)}")
    print(f"  - 每epoch批次数: {len(train_loader)}")
    print(f"  - 预计总epoch数: {cfg.total_step // len(train_loader)}")
    
    while should_keep_training:
        active_train_loader = train_loader

        model.train()
        if hasattr(model, 'module'):
            model.module.freeze_bn()
        else:
            model.freeze_bn()
            
        for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            _, left, right, disp_gt, valid = [x for x in data]
            with accelerator.autocast():
                disp_init_pred, disp_preds, depth_mono = model(left, right, iters=cfg.train_iters)
            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=cfg.max_disp)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metrics = accelerator.reduce(metrics, reduction='mean')
            accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics, total_step)

            # 可视化深度图和视差预测（针对SCARED数据集调整）
            if total_step % 20 == 0 and accelerator.is_main_process:
                image1_np = left[0].squeeze().cpu().numpy()
                image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                image1_np = image1_np.astype(np.uint8)
                image1_np = np.transpose(image1_np, (1, 2, 0))

                image2_np = right[0].squeeze().cpu().numpy()
                image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                image2_np = image2_np.astype(np.uint8)
                image2_np = np.transpose(image2_np, (1, 2, 0))

                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())

                accelerator.log({"scared/disp_pred": wandb.Image(disp_preds_np, caption="SCARED Prediction - step:{}".format(total_step))},
                                total_step)
                accelerator.log({"scared/disp_gt": wandb.Image(disp_gt_np, caption="SCARED GT - step:{}".format(total_step))}, total_step)
                accelerator.log({"scared/depth_mono": wandb.Image(depth_mono_np, caption="SCARED Mono Depth - step:{}".format(total_step))},
                                total_step)
                accelerator.log({"scared/left_img": wandb.Image(image1_np, caption="SCARED Left Image - step:{}".format(total_step))},
                                total_step)
                accelerator.log({"scared/right_img": wandb.Image(image2_np, caption="SCARED Right Image - step:{}".format(total_step))},
                                total_step)

            # 保存检查点
            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/scared_%d.pth' % (total_step))
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    print(f"Saved SCARED checkpoint at step {total_step}")
                    del model_save

            # 验证
            if (total_step > 0) and (total_step % cfg.val_frequency == 0):
                print(f"Running validation at step {total_step}...")
                model.eval()
                elem_num, total_epe, total_out, total_mae, total_d1, total_bad1, total_bad2, total_bad3 = 0, 0, 0, 0, 0, 0, 0, 0
                viz_done_in_epoch = False
                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                    _, left_orig, right_orig, disp_gt, valid = [x for x in data]
                    padder = InputPadder(left_orig.shape, divis_by=32)
                    left, right = padder.pad(left_orig, right_orig)
                    with torch.no_grad():
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    disp_pred = padder.unpad(disp_pred)
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                    if accelerator.is_main_process and not viz_done_in_epoch:
                        image1_np = left_orig[0].squeeze().cpu().numpy()
                        image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                        image1_np = image1_np.astype(np.uint8)
                        image1_np = np.transpose(image1_np, (1, 2, 0))

                        image2_np = right_orig[0].squeeze().cpu().numpy()
                        image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                        image2_np = image2_np.astype(np.uint8)
                        image2_np = np.transpose(image2_np, (1, 2, 0))

                        disp_pred_np_val = gray_2_colormap_np(disp_pred[0].squeeze())
                        disp_gt_np_val = gray_2_colormap_np(disp_gt[0].squeeze())

                        accelerator.log({"val_scared/disp_pred": wandb.Image(disp_pred_np_val, caption="SCARED Val Prediction - step:{}".format(total_step))},
                                        total_step)
                        accelerator.log({"val_scared/disp_gt": wandb.Image(disp_gt_np_val, caption="SCARED Val GT - step:{}".format(total_step))}, total_step)
                        accelerator.log({"val_scared/left_img": wandb.Image(image1_np, caption="SCARED Val Left Image - step:{}".format(total_step))},
                                        total_step)
                        accelerator.log({"val_scared/right_img": wandb.Image(image2_np, caption="SCARED Val Right Image - step:{}".format(total_step))},
                                        total_step)
                        
                        viz_done_in_epoch = True
                    
                    # 使用统一的disparity_epe函数计算EPE
                    # 确保输入是3D张量 (B, H, W)
                    disp_pred_3d = disp_pred.squeeze(1) if len(disp_pred.shape) == 4 else disp_pred
                    disp_gt_3d = disp_gt.squeeze(1) if len(disp_gt.shape) == 4 else disp_gt
                    epe = disparity_epe(disp_pred_3d, disp_gt_3d, max_disparity=cfg.max_disp)
                    
                    # 计算其他指标
                    abs_error = torch.abs(disp_pred - disp_gt)
                    out = (abs_error > 1.0).float()
                    mae = abs_error  # 计算MAE
                    
                    # 计算D1, bad1, bad2, bad3指标
                    d1_error = (abs_error > 1.0) & (abs_error > 0.05 * torch.abs(disp_gt))
                    bad1_error = abs_error > 1.0
                    bad2_error = abs_error > 2.0
                    bad3_error = abs_error > 3.0
                    
                    # 处理张量维度
                    out = torch.squeeze(out, dim=1)
                    mae = torch.squeeze(mae, dim=1)
                    d1_error = torch.squeeze(d1_error, dim=1)
                    bad1_error = torch.squeeze(bad1_error, dim=1)
                    bad2_error = torch.squeeze(bad2_error, dim=1)
                    bad3_error = torch.squeeze(bad3_error, dim=1)
                    disp_gt = torch.squeeze(disp_gt, dim=1)
                    
                    # 计算有效像素的指标
                    valid_mask = (valid >= 0.5) & (disp_gt.abs() < 192)
                    # EPE已经通过disparity_epe函数计算，直接使用
                    epe_val = epe
                    out_val = out[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(1.0, device=out.device)
                    mae_val = mae[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0, device=mae.device)
                    d1_val = d1_error[valid_mask].float().mean() if valid_mask.sum() > 0 else torch.tensor(1.0, device=d1_error.device)
                    bad1_val = bad1_error[valid_mask].float().mean() if valid_mask.sum() > 0 else torch.tensor(1.0, device=bad1_error.device)
                    bad2_val = bad2_error[valid_mask].float().mean() if valid_mask.sum() > 0 else torch.tensor(1.0, device=bad2_error.device)
                    bad3_val = bad3_error[valid_mask].float().mean() if valid_mask.sum() > 0 else torch.tensor(1.0, device=bad3_error.device)
                    
                    epe, out, mae, d1, bad1, bad2, bad3 = accelerator.gather_for_metrics((epe_val, out_val, mae_val, d1_val, bad1_val, bad2_val, bad3_val))
                    # 处理标量情况
                    if len(epe.shape) > 0:
                        elem_num += epe.shape[0]
                    else:
                        elem_num += 1

                    if len(epe.shape) > 0:
                        for i in range(epe.shape[0]):
                            total_epe += epe[i]
                            total_out += out[i]
                            total_mae += mae[i]
                            total_d1 += d1[i]
                            total_bad1 += bad1[i]
                            total_bad2 += bad2[i]
                            total_bad3 += bad3[i]
                    else:
                        total_epe += epe
                        total_out += out
                        total_mae += mae
                        total_d1 += d1
                        total_bad1 += bad1
                        total_bad2 += bad2
                        total_bad3 += bad3
                        
                    accelerator.log({
                        'val/epe': total_epe / elem_num, 
                        'val/mae': total_mae / elem_num, 
                        'val/d1': 100 * total_d1 / elem_num,
                        'val/bad1': 100 * total_bad1 / elem_num,
                        'val/bad2': 100 * total_bad2 / elem_num,
                        'val/bad3': 100 * total_bad3 / elem_num,
                        'val/d1_old': 100 * total_out / elem_num
                    }, total_step)
                    
                print(f"SCARED Validation - Step {total_step}:")
                print(f"  EPE: {total_epe / elem_num:.4f}, MAE: {total_mae / elem_num:.4f}")
                print(f"  D1: {100 * total_d1 / elem_num:.2f}%, Bad1: {100 * total_bad1 / elem_num:.2f}%")
                print(f"  Bad2: {100 * total_bad2 / elem_num:.2f}%, Bad3: {100 * total_bad3 / elem_num:.2f}%")

                model.train()
                if hasattr(model, 'module'):
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    # 保存最终模型
    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/scared_final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        print(f"Saved final SCARED model to {save_path}")
        del model_save

    accelerator.end_training()
    print("SCARED training completed!")


if __name__ == '__main__':
    main()
