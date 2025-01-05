#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

if 'META_INTERNAL' in os.environ.keys() and os.environ['META_INTERNAL'] == "False":
    generate_html = None
    from dust3r.dummy_io import *
else:
    from meta_internal.io import *
    from meta_internal.html_gen.run_model_doctor import generate_html

import argparse
import datetime
import json
import numpy as np
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

from dust3r.model import AsymmetricCroCo3DStereo, AsymmetricCroCo3DStereoMultiView, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
from dust3r.pcd_render import pcd_render, save_image_manifold, save_video_combined
from dust3r.gs import gs_render
from dust3r.utils.geometry import inv, geotrf

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

from torch.utils.data import default_collate

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
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--allow_first_test', default=1, type=int)
    parser.add_argument('--only_test', default=0, type=int)
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
    parser.add_argument('--miter', default=0, type=int,
                        help='No. of extra inference')
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
    set_device(args.gpu) # 0

    args.output_dir = get_log_dir_warp(args.output_dir)

    print("output_dir: "+args.output_dir)
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
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    train_epoch_size = real_batch_size * len(data_loader_train)
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {}
    for dataset_name in args.test_dataset.split('+'):
        dataset = build_dataset(dataset_name, args.batch_size, args.num_workers, test=True)
        dataset_name = dataset.dataset.tb_name
        data_loader_test[dataset_name] = dataset

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model) # 
    model_name = args.model.split('(')[0]
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
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
        print(f'Total number of parameters: {total_params}') # ≈1B 

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

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

    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)

    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs from {args.start_epoch}") # 
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs+1):

        t_save = -time.time()
    
        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch-1, 'last', best_so_far)
        t_save += time.time()

        # Test on multiple datasets
        new_best = False
        # if False:
        if ((epoch == 0 and args.allow_first_test > 0) or (epoch != 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0)) or epoch == 1:
            test_stats = {}
            test_set_id = -1
            for test_name, testset in data_loader_test.items():
                test_set_id += 1
                t_test = time.time()
                print('test name', test_name)
                stats = test_one_epoch(model, test_criterion, testset,
                                       device, epoch, train_epoch_size, log_writer=log_writer, args=args, prefix=test_name, miter = args.miter, test_set_id = test_set_id)
                test_stats[test_name] = stats

                # Save best of all
                if stats['loss_med'] < best_so_far:
                    best_so_far = stats['loss_med']
                    new_best = True
                print('test epoch time', epoch, time.time() - t_test)

        t_save -= time.time()
        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch-1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch-1, 'best', best_so_far)
        t_save += time.time()

        if epoch >= args.epochs or args.only_test:
            break  # exit after writing last test to disk

        # Train
        t_train = time.time()
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, train_epoch_size,
            log_writer=log_writer,
            args=args)
        print('train epoch time', epoch, time.time() - t_train)
        print('save epoch time', epoch, t_save)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint-final.pth')
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


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
            x_max = float(np.max(x_list, axis = 1).mean())
            x_min = float(np.min(x_list, axis = 1).mean())
            if k_base+'_first' not in ldk:
                loss_details[k_base+'_first'] = x_first
            # if k_base+'_best' not in ldk:
            loss_details[k_base+'_max'] = x_max
            loss_details[k_base+'_min'] = x_min

    return loss_details

def postprocess_batch(batch): # here the randomized number of inference views / number of rendered views are applied to the whole batch.

    nv, nr = batch[0]['random_nv_nr'][0].cpu().numpy() # we are always using the first sample's No. of views / No. of rendered views and apply it to all samples in the batch
    while len(batch) > nv:
        del batch[-1]
    batch = batch[:nv]
    ni = nv - nr
    for i in range(ni):
        batch[i]['only_render'][:] = False
    for i in range(ni, nv):
        batch[i]['only_render'][:] = True
    return batch, ni

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, epoch_size,
                    args,
                    log_writer=None):
    t_all = -time.time()
    assert torch.backends.cuda.matmul.allow_tf32 == True
    t_misc_1 = -time.time()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    t_misc_1 += time.time()

    t_misc_2 = -time.time()

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()
    t_misc_2 += time.time()
    t_misc_3 = 0
    t_misc_4 = 0

    t_inference = 0
    t_bp = 0

    print('before training')

    t_all_time = [time.time()]
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        batch, ni = postprocess_batch(batch)
        t_misc_3 -= time.time()
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)
        
        t_misc_3 += time.time()
        
        t_inference_i = -time.time()
        print('check sync train before foward', misc.get_rank(), epoch, data_iter_step)
        # torch.cuda.synchronize()
        # delta = 1
        # if epoch > 0:
        #     delta = 0
        delta = 1
        need_log = data_iter_step == 0 or ((data_iter_step + delta) % accum_iter == 0 and ((data_iter_step + delta) % (accum_iter * args.print_freq)) == 0)

        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp), ret='loss', log = need_log)
        # torch.cuda.synchronize()
        t_inference_i += time.time()
        t_inference += t_inference_i
        t_bp_i = -time.time()
        loss, loss_details = loss_tuple  # criterion returns two values

        print('check sync train after forward', misc.get_rank(), epoch, data_iter_step)
        loss /= accum_iter
        if loss > 10:
            print('strange loss appears', loss)
            loss = loss * 0.
        norm = loss_scaler(loss, optimizer, parameters=model.parameters(), # backward inside, no clip grad
                    update_grad=(data_iter_step + 1) % accum_iter == 0, model = model)
        if norm is not None and norm > 1000:
            print('strange norm appears', norm)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        t_bp_i += time.time()
        t_bp += t_bp_i
        t_all_time.append(time.time())
        print('train batch', 'step', data_iter_step, 'len data', len(data_loader), 'rank', misc.get_rank(), 'epoch_f', epoch_f, 'inference time', t_inference_i, t_inference / (1 + data_iter_step),  'bp time', t_bp_i, t_bp / (1 + data_iter_step), 'all time', t_all_time[-1] - t_all_time[-2], (t_all_time[-1] - t_all_time[0]) / (1 + data_iter_step))
        # inference time 0.23065853118896484 bp time 0.42483043670654297 # all time is similar for 4x8 and 1x8, which means 4x8 is indeed more efficient
        t_misc_4 -= time.time()
        lr = optimizer.param_groups[0]["lr"]
        for k in list(loss_details.keys()):
            if not isinstance(loss_details[k], (float, int)):
                loss_details.pop(k)
        if need_log:
            
            loss_value = float(loss * accum_iter)
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value), force=True)
                sys.exit(1)
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value, **loss_details)

        del loss
        del batch
        
        # print('train_loss debug', data_iter_step, accum_iter, data_iter_step, args.print_freq, ((data_iter_step + 1) % (accum_iter * args.print_freq)), log_writer)
        if need_log:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            # epoch_1000x = int(epoch_f * 1000)
            epoch_1000x = int(epoch_f * epoch_size)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            norm_item = norm.item() if norm is not None else 0.
            log_writer.add_scalar('train_grad_norm', norm_item, epoch_1000x)
            log_writer.add_scalar('train/time/all', t_all_time[-1] - t_all_time[-2], epoch_1000x)
            log_writer.add_scalar('train/time/ff', t_inference_i, epoch_1000x)
            log_writer.add_scalar('train/time/bp', t_bp_i, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_f * len(data_loader), epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_'+name, val, epoch_1000x)
        t_misc_4 += time.time()
    # gather the stats from all processes
    t_misc_5 = -time.time()
    metric_logger.synchronize_between_processes()
    t_misc_5 += time.time()

    t_all += time.time()
    print('train misc time', t_misc_1, t_misc_2, t_misc_3, t_misc_4, t_misc_5, t_all, t_inference, t_bp, t_all - t_inference - t_bp) # all miscs are very small train misc time 0.041296958923339844 0.0005085468292236328 0.002261638641357422 0.0012340545654296875 130.9805166721344
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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
        n_ref = all_info['pred1']['pts3d'].shape[0] // bs
        if n_ref != 1:
            from dust3r.losses import extend_gts
            views = [all_info['view1']] + all_info['view2s']
            views = extend_gts(views, n_ref, bs)
            all_info['view1'] = views[0]
            all_info['view2s'] = views[1:]
            bs = n_ref * bs

        for img_id in range(bs):
            # import fbvscode
            # fbvscode.set_trace()
            img_id_mref_first = img_id
            # img_id_mref_first = n_ref * img_id # 00022_id_000000001_test_dataName_hs_3.0_sceneName_Beach_refId_00_00000_0033_test
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
            conf = torch.cat([all_info['pred1']['conf'][img_id_mref_first].reshape(-1, 1)] + [x['conf'][img_id_mref_first].reshape(-1, 1) for x in all_info['pred2s']], 0) # [N, 1]
            conf_sorted = conf.reshape(-1).sort()[0]
            conf_thres = float(conf_sorted[int(conf.shape[0] * 0.03)])
            cam1 = all_info['view1']['camera_pose'][img_id] # c2w
            pts3d = geotrf(cam1, pts3d)  # B,H,W,3
            # img_id_name = str(img_id).zfill(3)
            img_id_name = str(time.time()).split('.')[1]
            img_id_name = f"nref_{img_id % n_ref}_{str(time.time()).split('.')[1]}"
            video_pcd_gt =      pcd_render(pts3d_gt, rgb, tgt = None, normalize = True)
            video_pcd =         pcd_render(pts3d   , rgb, tgt = None, normalize = True)
            video_pcd_conf =    pcd_render(pts3d   , rgb, tgt = None, normalize = True, mask = (conf > conf_thres) * valid_masks.reshape(-1, 1)) # log(3)
            print('vis conf range', conf.min(), conf.mean(), conf.max(), conf_thres, (conf < 1.02).float().mean(), (conf < 1.03).float().mean(), (conf < 1.06).float().mean(), (conf < 1.09).float().mean())
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
    else:
        for img_id in range(bs):
            rgb1 = all_info['view1']['img'][img_id].permute(1,2,0)
            rgb2 = all_info['view2']['img'][img_id].permute(1,2,0)
            rgb      = torch.cat([rgb1.reshape(-1, 3), rgb2.reshape(-1, 3)], 0)
            pts3d_gt = torch.cat([all_info['view1']['pts3d'][img_id].reshape(-1, 3), all_info['view2']['pts3d'][img_id].reshape(-1, 3)], 0)
            pts3d    = torch.cat([all_info['pred1']['pts3d'][img_id].reshape(-1, 3), all_info['pred2']['pts3d_in_other_view'][img_id].reshape(-1, 3)], 0)

            cam1 = all_info['view1']['camera_pose'][img_id] # c2w -> w2c
            pts3d = geotrf(cam1, pts3d)  # B,H,W,3
            
            img_id_name = str(img_id).zfill(3)
            pcd_render(pts3d_gt, rgb, f"{args.output_dir}/videos/{name}_{img_id_name}_gt.mp4", normalize = True)
            pcd_render(pts3d   , rgb, f"{args.output_dir}/videos/{name}_{img_id_name}.mp4", normalize = True)
            torch.save(loss_and_others, f"{args.output_dir}/results/{name}_{img_id_name}.pth")
            # rgb is -1~1, shape = (res,res,3)
            save_image_manifold(((rgb1 + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_rgb1.png")
            save_image_manifold(((rgb2 + 1) / 2 * 255).cpu().numpy().astype(np.uint8), f"{args.output_dir}/videos/{name}_{img_id_name}_rgb2.png")

def to_device(data, device):

    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(element, device) for element in data]
    elif isinstance(data, tuple):
        return tuple(to_device(element, device) for element in data)
    else:
        return data

def update_batch(batch, loss_and_others, data_loader):
    views = [loss_and_others['view1']] + loss_and_others['view2s']
    ids = views[0]['idx'][0]
    pts_pred = [loss_and_others['pred1']['pts3d']] + [x['pts3d_in_other_view'] for x in loss_and_others['pred2s']] # [bs, res, res, 3] each
    pts_pred = torch.stack(pts_pred, dim = 1) # [bs, n_inference, res, res, 3]
    pts_pred_center_view = pts_pred.mean(dim = (2,3)) # [bs, n_inference, 3]
    pts_pred_center = pts_pred_center_view.mean(dim = 1) # [bs, 3]
    view_dis = torch.norm(pts_pred_center_view - pts_pred_center.unsqueeze(1), dim = 2) # [bs, n_inference]
    nearest_view_id = view_dis.argmin(dim = 1) # [bs]
    new_batch = [data_loader.dataset.__getitem_bsvd__(x.item(), y.item()) for x, y in zip(ids.long(), nearest_view_id)]
    new_batch = default_collate(new_batch)
    return new_batch

@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   train_epoch_size, args, log_writer=None, prefix='test', miter = False, test_set_id = 0):
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
    for batch_id, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        batch, ni = postprocess_batch(batch)
        t = time.time()
        # print('test batch 1st', batch_id, len(data_loader), epoch, misc.get_rank())
        # torch.cuda.synchronize()
        t_inference -= time.time()
        if miter:
            loss_and_others = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp), ret=None)
            batch = update_batch(batch, loss_and_others, data_loader)
            # print('pts3d_2_avg before', loss_and_others['loss'][1]['Regr3D_ScaleShiftInv_pts3d_2'])
            # import fbvscode
            # fbvscode.set_trace()
        loss_and_others = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp), ret=None)
        # print('pts3d_2_avg after', loss_and_others['loss'][1]['Regr3D_ScaleShiftInv_pts3d_2'])
        # torch.cuda.synchronize()
        t_inference += time.time()
        # print('test batch 2nd', batch_id, len(data_loader), epoch, misc.get_rank(), 'inference time', time.time() - t, batch[0]['pts3d'].shape)

        t_save -= time.time()
        # print('data_loader', type(data_loader.dataset).__name__)
        if data_loader.dataset.save_results:
            global_rank = misc.get_rank()
            prefix_save = [str(epoch).zfill(5) + "_testSetID_" + str(test_set_id).zfill(3)]
            # prefix_save = [str(epoch).zfill(5), str(batch_id).zfill(5), str(global_rank).zfill(4), data_loader.dataset.save_prefix]
            save_results(loss_and_others, batch, prefix_save, args)
        t_save += time.time()
        # print('test batch 3rd', batch_id, len(data_loader), epoch, misc.get_rank())
        loss_tuple = loss_and_others['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values
        n_ref = int(loss_details['n_ref'])
        loss_details.pop('n_ref')
        loss_details = add_first_best(loss_details, n_ref)
        for k in list(loss_details.keys()):
            if not isinstance(loss_details[k], (float, int)):
                loss_details.pop(k)
        metric_logger.update(loss=float(loss_value), **loss_details)
        # if batch_id >= 1:
        #     break

    t_batch += time.time()
    # gather the stats from all processes
    t_log = - time.time()
    
    if data_loader.dataset.save_results and misc.get_rank() == 0:
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
            log_writer.add_scalar(prefix+'/'+name, val, epoch_1000x)
    t_log += time.time()
    print('test all time', prefix, 'batch', t_batch, t_batch - t_inference - t_save, 'inference', t_inference, 'save', t_save, 'log', t_log, 'two begins', t_begin1, t_begin2) # inference and log is small, batch is kind of large, but  
    # test all time  100 @ ScannetPair_test batch 70.40310192108154 inference 5.6025426387786865 save 0.0006468296051025391 log 0.0017290115356445312
    # seems batch cost a lot of time, maybe from dataloading? testing now, inference is fast, save cost time in visualization but not torch.save, t_log and t_begin is fast.
    return results


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
