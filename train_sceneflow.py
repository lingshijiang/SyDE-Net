import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
# from util import InputPadder
from core.utils.utils import InputPadder
from core.monster import Monster 
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pathlib import Path

def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }
    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0}, 
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')


    return optimizer, scheduler

@hydra.main(version_base=None, config_path='config', config_name='train_sceneflow')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb', kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs={'wandb': cfg.wandb})

    train_dataset = datasets.fetch_dataloader(cfg)

    aug_params = {}

    if cfg.total_step == 1:  # 如果只训练一步
        # 只加载一个批次
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size // cfg.num_gpu,
                                                   pin_memory=True, shuffle=False, num_workers=int(4), drop_last=True,
                                                   sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=1))
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size // cfg.num_gpu,
                                                   pin_memory=True, shuffle=True, num_workers=int(4), drop_last=True)
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    model = Monster(cfg)
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

        model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device)

    total_step = 0
    should_keep_training = True
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

            ####visualize the depth_mono and disp_preds
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
                
                accelerator.log({"disp_pred": wandb.Image(disp_preds_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"disp_gt": wandb.Image(disp_gt_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"depth_mono": wandb.Image(depth_mono_np, caption="step:{}".format(total_step))}, total_step)

            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    del model_save

                    if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                        if accelerator.is_main_process:
                            save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
                            model_save = accelerator.unwrap_model(model)
                            torch.save(model_save.state_dict(), save_path)
                            del model_save

                    if (total_step > 0) and (total_step % cfg.val_frequency == 0):
                        print(f"\n=========== Starting validation at step {total_step} ===========")
                        model.eval()
                        elem_num, total_epe, total_out = 0, 0, 0
                        valid_samples = 0  # 记录成功处理的样本数

                        max_val_samples = getattr(cfg, 'val_samples', len(val_dataset))
                        # 如果指定了较小的样本数量，创建一个子集
                        if max_val_samples < len(val_dataset):
                            # 随机选择指定数量的样本索引
                            all_indices = list(range(len(val_loader.dataset)))
                            selected_indices = torch.randperm(len(all_indices))[:max_val_samples].tolist()

                            indices = torch.randperm(len(val_dataset))[:max_val_samples].tolist()

                            # 使用PyTorch的Subset类创建子集
                            val_dataset = torch.utils.data.Subset(val_dataset, indices)
                            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(1),
                                                                     pin_memory=True, shuffle=False, num_workers=int(4),
                                                                     drop_last=False)
                        else:
                            val_loader = val_loader
                        for batch_idx, data in enumerate(
                                tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process)):
                            try:  # 打印当前处理的样本编号
                                # 添加批次索引信息
                                print(f"\n===== Processing validation sample {batch_idx} =====")
                                # _, left, right, disp_gt, valid = [x for x in data]
                                _, left, right, disp_gt, valid, disp_gt_right, valid_right = [x for x in data]
                                left, right = left.to(accelerator.device), right.to(accelerator.device)
                                disp_gt, valid = disp_gt.to(accelerator.device), valid.to(accelerator.device)
                                # 检查输入数据
                                if batch_idx % 100 == 0:
                                    print(f"Processing validation sample {batch_idx}")
                                padder = InputPadder(left.shape, divis_by=32)
                                left, right = padder.pad(left, right)
                                with torch.no_grad():
                                    disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                                disp_pred = padder.unpad(disp_pred)
                                assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                                epe = torch.abs(disp_pred - disp_gt)
                                out = (epe > 1.0).float()
                                epe = torch.squeeze(epe, dim=1)
                                out = torch.squeeze(out, dim=1)
                                disp_gt = torch.squeeze(disp_gt, dim=1)
                                # 创建有效像素掩码
                                valid_mask = (valid >= 0.5) & (disp_gt.abs() < 192)
                                # print(f"  - valid_mask: {valid_mask.shape}")
                                valid_pixels = valid_mask.sum().item()
                                if valid_pixels == 0:
                                    print(f"  WARNING: Sample {batch_idx} has no valid pixels after filtering!")
                                    continue
                                # 先获取有效像素的均值，避免计算空张量的均值
                                sample_epe_mean = epe[valid_mask].mean()
                                sample_out_mean = out[valid_mask].mean()
                                # 使用accelerator收集指标
                                epe_val, out_val = accelerator.gather_for_metrics((sample_epe_mean, sample_out_mean))
                                # 安全处理收集的指标
                                if isinstance(epe_val, torch.Tensor):
                                    if len(epe_val.shape) > 0:  # 张量有维度
                                        elem_num += epe_val.shape[0]
                                        for i in range(epe_val.shape[0]):
                                            if not torch.isnan(epe_val[i]).item() and not torch.isinf(
                                                    epe_val[i]).item():
                                                total_epe += epe_val[i]
                                                total_out += out_val[i]
                                    else:  # 0维张量 (标量)
                                        elem_num += 1
                                        if not torch.isnan(epe_val).item() and not torch.isinf(epe_val).item():
                                            total_epe += epe_val
                                            total_out += out_val
                                else:  # 非张量对象
                                    elem_num += 1
                                    if not np.isnan(epe_val) and not np.isinf(epe_val):
                                        total_epe += epe_val
                                        total_out += out_val

                                valid_samples += 1

                                # 每处理50个样本打印一次中间结果
                                if (batch_idx + 1) % 50 == 0 and elem_num > 0:
                                    print(
                                        f"Intermediate results after {batch_idx + 1} samples: EPE={total_epe / elem_num:.4f}, D1={100 * total_out / elem_num:.2f}%")

                            except Exception as e:
                                import traceback
                                print(f"ERROR processing validation sample {batch_idx}:")
                                print(traceback.format_exc())

                        print(f"\nValidation summary:")
                        print(f"  Total samples: {len(val_loader)}")
                        print(
                            f"  Successfully processed: {valid_samples} ({valid_samples / len(val_loader) * 100:.2f}%)")
                        # 计算并记录最终结果
                        if elem_num > 0:
                            mean_epe = total_epe / elem_num
                            mean_d1 = 100 * total_out / elem_num
                            print(f"  Final mean EPE: {mean_epe:.4f}, mean D1: {mean_d1:.2f}%")
                            accelerator.log({'val/epe': mean_epe, 'val/d1': mean_d1}, total_step)
                        else:
                            print("  WARNING: No valid elements for computing validation metrics!")
                            # 记录警告而非NaN
                            accelerator.log({'val/warning': 'No valid elements for metrics'}, total_step)
                        model.train()
                        # model.module.freeze_bn()
                        if hasattr(model, 'module'):
                            model.module.freeze_bn()
                        else:
                            model.freeze_bn()

                    if total_step == cfg.total_step:
                        should_keep_training = False
                        break

            if accelerator.is_main_process:
                save_path = Path(cfg.save_path + '/final.pth')
                model_save = accelerator.unwrap_model(model)
                torch.save(model_save.state_dict(), save_path)
                del model_save

            accelerator.end_training()

if __name__ == '__main__':
    main()