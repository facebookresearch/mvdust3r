# Copyright (C) 2025-present Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC BY-NC 4.0 (non-commercial use only).


import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
import imageio


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

if 'META_INTERNAL' in os.environ.keys() and os.environ['META_INTERNAL'] == "False":
    generate_html = None
    from dust3r.dummy_io import *
else:
    from meta_internal.io import *
    from meta_internal.html_gen.run_model_doctor import generate_html

from dust3r.model import AsymmetricCroCo3DStereo, AsymmetricCroCo3DStereoMultiView, inf 
import dust3r.utils.path_to_croco  # noqa: F401
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch # noqa
from inference_global_optimization import loss_of_one_batch_go_mv  # noqa
from dust3r.pcd_render import pcd_render, save_image_manifold, save_video_combined
from dust3r.gs import gs_render
from dust3r.utils.geometry import inv, geotrf

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', required=True, type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")

    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=20, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')

    # output dir
    parser.add_argument('--output_dir', default=None, type=str, help="path where to save the output")
    return parser


def main(args):
    print('args', args)
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    real_batch_size = args.batch_size * world_size
    print('world size', world_size, 'global_rank', global_rank, 'real_batch_size', real_batch_size)
    set_device(args.gpu)

    args.output_dir = get_log_dir_warp(args.output_dir)

    print("output_dir: "+args.output_dir) # manifold://ondevice_ai_writedata/tree/zgtang/dust3r/logs/torchx-dust3r_train-temp3
    if args.output_dir:
        g_pathmgr.mkdirs(args.output_dir)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if g_pathmgr.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    # data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    # train_epoch_size = real_batch_size * len(data_loader_train)
    train_epoch_size = real_batch_size * 100000
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {}
    for dataset_name in args.test_dataset.split('+'):
        dataset = build_dataset(dataset_name, args.batch_size, args.num_workers, test=True)
        dataset_name = dataset.dataset.tb_name
        data_loader_test[dataset_name] = dataset

    # data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
    #                     for dataset in args.test_dataset.split('+')}

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model) # 
    model_name = args.model.split('(')[0]
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device) # ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        model_loaded = eval(model_name).from_pretrained(get_local_path(args.pretrained)).to(device)
        print('Loading pretrained: ', args.pretrained, model_name) # 

        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=False)
        model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module
        total_params = sum(p.numel() for p in model_without_ddp.parameters())
        print(f'Total number of parameters: {total_params}') # 0.5B

    # following timm: set wd as 0 for bias and norm layers

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update({test_name+'_'+k: v for k, v in test_stats[test_name].items()})

            with g_pathmgr.open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")



    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    epoch = 0
    test_stats = {}
    test_set_id = -1
    for test_name, testset in data_loader_test.items():
        test_set_id += 1
        t_test = time.time()
        print('test name', test_name)
        stats = test_one_epoch(model, test_criterion, testset,
                                device, epoch, train_epoch_size, log_writer=log_writer, args=args, prefix=test_name, test_set_id = test_set_id)
        test_stats[test_name] = stats

        print('test epoch time', time.time() - t_test)

    # Save more stuff
    write_log_stats(epoch, train_stats, test_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader

def save_results(loss_and_others, batch, name_list, args):
    all_info = loss_and_others
    other_info = loss_and_others['loss'][1]
    # view1: img (real_bs * 2 (data aug for symmetry), 3, res=224, res), depthmap, camera_pose (real_bs * 2, 4, 4), camera_intrinsics, dataset, label, instance, idx, true_shape, pts3d (real_bs * 2, res, res, 3), valid_mask, rng
    # pred1: pts3d, conf
    # pred2: pts3d_in_other_view, conf
    g_pathmgr.mkdirs(args.output_dir + '/results')
    g_pathmgr.mkdirs(args.output_dir + '/videos')
    bs = all_info['view1']['img'].shape[0] # real_bs * 2 = bs
    
    if 'view2s' in all_info.keys(): # MV here
    
        for img_id in range(bs):
            img_id_mref_first = img_id
            n_ref = 1
            # img_id_mref_first = n_ref * img_id
            label = batch[0]['label'][img_id // n_ref]
            name = "_".join(name_list[0:1] + [label] + name_list[1:])
            rgb1 = all_info['view1']['img'][img_id].permute(1,2,0)
            valid_mask1 = all_info['view1']['valid_mask'][img_id].reshape(-1)
            num_render_views = all_info['view2s'][0].get("num_render_views", torch.zeros([0]).long())[0].item()
            rgb2s_all = [x['img'][img_id].permute(1,2,0) for x in all_info['view2s']]
            valid_mask2s = [x['valid_mask'][img_id].reshape(-1) for x in all_info['view2s']]
            rgb2s = rgb2s_all[:-num_render_views] if num_render_views else rgb2s_all
            valid_mask2s = valid_mask2s[:-num_render_views] if num_render_views else valid_mask2s
            rgb      = torch.cat([rgb1.reshape(-1, 3)] + [rgb2.reshape(-1, 3) for rgb2 in rgb2s], 0)
            valid_masks = torch.stack([valid_mask1] + valid_mask2s, 0)
            pts3d_gt = torch.cat([all_info['view1']['pts3d'][img_id].reshape(-1, 3)] + [x['pts3d'][img_id].reshape(-1, 3) for x in (all_info['view2s'][:-num_render_views] if num_render_views else all_info['view2s'])], 0)
            pts3d    = torch.cat([all_info['pred1']['pts3d'][img_id_mref_first].reshape(-1, 3)] + [x['pts3d_in_other_view'][img_id_mref_first].reshape(-1, 3) for x in all_info['pred2s']], 0)
            conf = torch.cat([all_info['pred1']['conf'][img_id_mref_first].reshape(-1, 1)] + [x['conf'][img_id_mref_first].reshape(-1, 1) for x in all_info['pred2s']], 0)
            conf_sorted = conf.reshape(-1).sort()[0]
            conf_thres = float(conf_sorted[int(conf.shape[0] * 0.03)])
            # conf_thres = 0.5
            cam1 = all_info['view1']['camera_pose'][img_id] # c2w
            pts3d = geotrf(cam1, pts3d)  # B,H,W,3
            # img_id_name = str(img_id).zfill(3)
            # import fbvscode
            # fbvscode.set_trace()
            img_id_name = f"nref_{img_id % n_ref}_{str(time.time()).split('.')[1]}"
            video_pcd_gt =      pcd_render(pts3d_gt, rgb, tgt = None, normalize = True)
            video_pcd =         pcd_render(pts3d   , rgb, tgt = None, normalize = True)
            # video_pcd_conf = video_pcd
            video_pcd_conf =    pcd_render(pts3d   , rgb, tgt = None, normalize = True, mask = conf > conf_thres * valid_masks.reshape(-1, 1)) # log(3)
            # print('vis conf range', conf.min(), conf.mean(), conf.max(), conf_thres, (conf < 1.02).float().mean(), (conf < 1.03).float().mean(), (conf < 1.06).float().mean(), (conf < 1.09).float().mean())
            save_video_combined([video_pcd, video_pcd_conf, video_pcd_gt], f"{args.output_dir}/videos/{name}_{img_id_name}_and_gt.mp4")
            if 'scale' in all_info['pred1'].keys(): # 3DGS predicted
                gts = [all_info['view1']] + [v for v in (all_info['view2s'][:-num_render_views] if num_render_views else all_info['view2s'])]
                preds = [all_info['pred1']] + [v for v in all_info['pred2s']]
                video_gs_gt = gs_render(gts, preds, img_id, img_id_mref_first, cam1, normalize = True, gt_pcd = True, gt_img = True)
                video_gs_gt_img_only = gs_render(gts, preds, img_id, img_id_mref_first, cam1, normalize = True, gt_pcd = False, gt_img = True)
                video_gs =    gs_render(gts, preds, img_id, img_id_mref_first, cam1, normalize = True)
                save_video_combined([video_gs, video_gs_gt_img_only, video_gs_gt], f"{args.output_dir}/videos/{name}_{img_id_name}_and_gt_GS.mp4")
            # import fbvscode
            # fbvscode.set_trace()
            other_info_web = {k: float(other_info[k][img_id_mref_first]) for k in other_info.keys() if "_list" in k}
            torch.save(other_info_web, f"{args.output_dir}/videos/{name}_{img_id_name}.pth")
            # rgb is -1~1, shape = (res,res,3)
            rgbs = [rgb1]
            save_image_manifold(((rgb1 + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_rgb1.png")
            for rgb_id, rgb2 in enumerate(rgb2s_all):
                rgbs.append(rgb2)
                save_image_manifold(((rgb2 + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_rgb{rgb_id + 2}.png")
            rgbs = torch.cat(rgbs, dim = 1) # [h,w (combine here),3]
            save_image_manifold(((rgbs + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_rgb_all.png")
            if "render_all" in other_info.keys():
                render_all = other_info["render_all"] # render_all[img_id]: [nv, 224, 224, 3]
                save_image_manifold(((render_all[img_id_mref_first].permute(1,0,2,3).flatten(1,2) + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_gs.png")
            if "render_relocated_all" in other_info.keys():
                render_relocated_all = other_info["render_relocated_all"] # render_all[img_id]: [nv, 224, 224, 3]
                save_image_manifold(((render_relocated_all[img_id_mref_first].permute(1,0,2,3).flatten(1,2) + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_gs_relocated.png")
        # for img_id in range(bs):
        #     label = batch[0]['label'][img_id]
        #     name = "_".join(name_list[0:1] + [label] + name_list[1:])
        #     rgb1 = all_info['view1']['img'][img_id].permute(1,2,0)
        #     rgb2s = [x['img'][img_id].permute(1,2,0) for x in all_info['view2s']]
        #     rgb      = torch.cat([rgb1.reshape(-1, 3)] + [rgb2.reshape(-1, 3) for rgb2 in rgb2s], 0)
        #     pts3d_gt = torch.cat([all_info['view1']['pts3d'][img_id].reshape(-1, 3)] + [x['pts3d'][img_id].reshape(-1, 3) for x in all_info['view2s']], 0)
        #     pts3d    = torch.cat([all_info['pred1']['pts3d'][img_id].reshape(-1, 3)] + [x['pts3d_in_other_view'][img_id].reshape(-1, 3) for x in all_info['pred2s']], 0)

        #     cam1 = all_info['view1']['camera_pose'][img_id] # c2w -> w2c
        #     pts3d = geotrf(cam1, pts3d)  # B,H,W,3
        #     img_id_name = str(img_id).zfill(3)
        #     video_pcd_gt = pcd_render(pts3d_gt, rgb, tgt = None, normalize = True)
        #     video_pcd =    pcd_render(pts3d   , rgb, tgt = None, normalize = True)
        #     save_video_combined([video_pcd, video_pcd_gt], f"{args.output_dir}/videos/{name}_{img_id_name}_and_gt.mp4")
        #     other_info_web = {k: float(other_info[k][img_id]) for k in other_info.keys() if "_list" in k}
        #     torch.save(other_info_web, f"{args.output_dir}/videos/{name}_{img_id_name}.pth")

        #     # rgb is -1~1, shape = (res,res,3)
        #     rgbs = [rgb1]
        #     save_image_manifold(((rgb1 + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_rgb1.png")
        #     for rgb_id, rgb2 in enumerate(rgb2s):
        #         rgbs.append(rgb2)
        #         save_image_manifold(((rgb2 + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_rgb{rgb_id + 2}.png")
        #     rgbs = torch.cat(rgbs, dim = 1) # [h,w (combine here),3]
        #     save_image_manifold(((rgbs + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_rgb_all.png")
            
    else:
        raise NotImplementedError

def add_first_best(loss_details, n_ref):

    # import fbvscode
    # fbvscode.set_trace()
    ldk = list(loss_details.keys())
    for k in ldk:
        if k == 'loss':
            continue
        if "_list" in k:
            x_list = np.array(loss_details[k])
            k_base = k.replace('_list', '')
            x_list = x_list.reshape(-1, n_ref)
            x_first = float(x_list[:, 0].mean())
            x_best = float(np.max(x_list, axis = 1).mean())
            if k_base+'_first' not in ldk:
                loss_details[k_base+'_first'] = x_first
            if k_base+'_best' not in ldk:
                loss_details[k_base+'_best'] = x_best
    return loss_details

def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   train_epoch_size, args, log_writer=None, prefix='test', test_set_id = 0):
    t_begin1 = -time.time()
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    t_begin1 += time.time()
    t_begin2 = -time.time()
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        print('set in dataset')
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        print('set in sampler')
        data_loader.sampler.set_epoch(epoch)
    t_begin2 += time.time()
    t_batch = -time.time()
    t_inference = 0
    t_save = 0
    t1_sum = 0.
    t2_sum = 0.
    for batch_id, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        t = time.time()
        torch.cuda.synchronize()
        t_inference -= time.time()
        # loss_and_others = loss_of_one_batch(batch, model, criterion, device,
        #                                symmetrize_batch=True,
        #                                use_amp=bool(args.amp), ret=None)
        
        loss_and_others, t1, t2, n_v = loss_of_one_batch_go_mv(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp), ret=None)
        t1_sum += t1
        t2_sum += t2
        print('GO time', t1_sum / (batch_id + 1), t2_sum / (batch_id + 1), n_v)
        torch.cuda.synchronize()
        t_inference += time.time()
        print('test batch', batch_id, len(data_loader), 'time', time.time() - t, 'pts3d shape', batch[0]['pts3d'].shape)

        t_save -= time.time()
        print('data_loader', type(data_loader.dataset).__name__, batch[0]['label'][0])
        if data_loader.dataset.save_results:
            global_rank = misc.get_rank()
            prefix_save = [str(epoch).zfill(5) + "_testSetID_" + str(test_set_id).zfill(3)]
            save_results(loss_and_others, batch, prefix_save, args)
        t_save += time.time()

        loss_tuple = loss_and_others['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values
        n_ref = int(loss_details['n_ref'])
        loss_details.pop('n_ref')
        loss_details = add_first_best(loss_details, n_ref)
        
        for k in list(loss_details.keys()):
            if not isinstance(loss_details[k], (float, int)):
                loss_details.pop(k)
        # import fbvscode
        # fbvscode.set_trace()
        metric_logger.update(loss=float(loss_value), **loss_details)
        print('loss details', loss_details)

    t_batch += time.time()
    # gather the stats from all processes
    t_log = - time.time()
    
    if data_loader.dataset.save_results:
        if generate_html is not None:
            generate_html(args.output_dir + '/videos', args.output_dir + '/html')

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            # epoch_1000x = int(epoch * 1000)
            epoch_1000x = int(epoch * train_epoch_size)
            log_writer.add_scalar(prefix+'_'+name, val, epoch_1000x)
    t_log += time.time()
    print('test all time', prefix, 'batch', t_batch, t_batch - t_inference - t_save, 'inference', t_inference, 'save', t_save, 'log', t_log, 'two begins', t_begin1, t_begin2) # inference and log is small, batch is kind of large, but  
    # test all time  100 @ ScannetPair_test batch 70.40310192108154 inference 5.6025426387786865 save 0.0006468296051025391 log 0.0017290115356445312
    # seems batch cost a lot of time, maybe from dataloading? testing now, inference is fast, save cost time in visualization but not torch.save, t_log and t_begin is fast.
    return results


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
