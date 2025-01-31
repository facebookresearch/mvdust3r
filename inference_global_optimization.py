# Copyright (C) 2025-present Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC BY-NC 4.0 (non-commercial use only).


import argparse
import math

import os
import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
import time
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

from dust3r.pcd_render import pcd_render

def loss_of_one_batch_go_mv(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    views = batch
    view1, view2s = views[0], views[1:]
    for view in batch:
        for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    t1s, t2s = [], []
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        # pred1, pred2 = model(view1, view2s[0]) # pred1 pcd torch.Size([2, 224, 224, 3])
        # print('views img', view1['img'].max(), view1['img'].min(), view1['img'].shape) # views img tensor(1., device='cuda:0') tensor(-0.9216, device='cuda:0') torch.Size([bs, 3, 224, 224])
        # print(view1['img'].dtype) # float32
        # import fbvscode
        # fbvscode.set_trace()
        bs = view1['img'].shape[0]
        n_v_real = 1
        for view2_id, view2 in enumerate(view2s):
            if view2['only_render'][0].item():
                break
            n_v_real += 1
            
        view2s_all = view2s
        view2s = view2s[:n_v_real - 1]
        views = [view1] + view2s
        n_v = len(view2s) + 1
        # print('pred1 pcd', pred1['pts3d'].shape)
        # print('view1 pcd', view1['pts3d'].shape) # torch.Size([bs, 224, 224, 3])
        preds = [{'pts3d':[], 'conf':[], 'c2ws_pred':[], 'intrinsics_pred':[]} for i in range(n_v)]
        for i in range(bs):
            # print('camera pose shape', view1['camera_pose'].shape)
            pts3ds, c2ws, intrinsics, confs, t1, t2 = inference_global_optimization(model, device, False, [view1['img'][i]] + [view2['img'][i] for view2 in view2s], view1['camera_pose'][i])
            print('GO per scene time', t1, t2, n_v_real)
            t1s.append(t1)
            t2s.append(t2)
            for j in range(n_v):
                preds[j]['pts3d'].append(pts3ds[j])
                preds[j]['conf'].append(confs[j])
                preds[j]['c2ws_pred'].append(c2ws[j])
                preds[j]['intrinsics_pred'].append(intrinsics[j])

        # pred1 : ['conf', 'rgb', 'opacity', 'scale', 'rotation', 'pts3d'] 
        # ('pts3d', torch.Size([bs, 224, 224, 3]), 3.4764482975006104, -1.5572370290756226), 
        # ('conf', torch.Size([bs, 224, 224]), 41.92277908325195, 1.0040476322174072)
        # ('rgb', torch.Size([bs, 224, 224, 3]), 0.8159868121147156, -0.8702595829963684)
        # ('opacity', torch.Size([bs, 224, 224, 1]), 0.999699592590332, 7.182779518188909e-05)
        # ('scale', torch.Size([bs, 224, 224, 3]), 0.03545345366001129, -0.04244176670908928), 
        # ('rotation', torch.Size([bs, 224, 224, 4]), 0.9999783039093018, -0.9999967813491821)
        for pred, view in zip(preds, views):
            pred['pts3d'] = torch.stack(pred['pts3d'], dim=0).detach()
            pred['conf'] = torch.stack(pred['conf'], dim=0).detach()
            # import fbvscode
            # fbvscode.set_trace()
            pred['c2ws_pred'] = torch.stack(pred['c2ws_pred'], dim=0).detach()
            pred['intrinsics_pred'] = torch.stack(pred['intrinsics_pred'], dim=0).detach()
            # pred['conf'] = pred['conf'].unsqueeze(-1)
            pred['rgb'] = view['img'].permute(0, 2, 3, 1)
            pred['opacity'] = torch.ones_like(pred['rgb'][:,:,:,0:1])
            
            for b in range(bs):
                conf_b = pred['conf'][b].reshape(-1)
                conf_sorted = conf_b.sort()[0]
                conf_thres = float(conf_sorted[int(conf_b.shape[0] * 0.03)])
                conf_mask = pred['conf'][b] < conf_thres # [224, 224]
                # print('conf_mask', conf_mask.float().mean())
                pred['opacity'][b][conf_mask] = 0
            
            pred['scale'] = torch.ones_like(pred['rgb']) * 1e-3 * 2
            pred['rotation'] = torch.ones_like(pred['rgb'][:,:,:,0:1].repeat(1,1,1,4))
            # print('preds', pred['pts3d'].shape) # [bs, 224, 224, 3]
        for pred in preds[1:]:
            pred['pts3d_in_other_view'] = pred.pop('pts3d')
        pred1, pred2s = preds[0], preds[1:]
        # loss is supposed to be symmetric
        # pred1, pred2 = model(view1, view2s[0]) # pred1 pcd torch.Size([2, 224, 224, 3])
        # pred2s = [pred2, pred2, pred2]
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2s_all, pred1, pred2s, log = True) if criterion is not None else None
    # print('in go_mv all keys')
    # print('views', [k for k in view1.keys()], [[k for k in view2.keys()] for view2 in view2s])
    # print('preds', [k for k in pred1.keys()], [[k for k in pred2.keys()] for pred2 in pred2s])
    # views ['img', 'depthmap', 'camera_pose', 'camera_intrinsics', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'pts3d', 'valid_mask', 'rng'] [['img', 'depthmap', 'camera_pose', 'camera_intrinsics', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'pts3d', 'valid_mask', 'rng'], ['img', 'depthmap', 'camera_pose', 'camera_intrinsics', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'pts3d', 'valid_mask', 'rng'], ['img', 'depthmap', 'camera_pose', 'camera_intrinsics', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'pts3d', 'valid_mask', 'rng']]
    # preds ['pts3d', 'conf'] [['conf', 'pts3d_in_other_view'], ['conf', 'pts3d_in_other_view'], ['conf', 'pts3d_in_other_view']]
    # import fbvscode
    # fbvscode.set_trace()
    view2s = batch[1:]
    result = dict(view1=view1, view2s=view2s, pred1=pred1, pred2s=pred2s, loss=loss)
    res = result[ret] if ret else result
    return res, float(np.mean(t1)), float(np.mean(t2)), n_v_real


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                         "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                         "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    schedule = "linear"
    niter = 300
    min_conf_thr = 3
    as_pointcloud = True
    mask_sky = False
    clean_depth = False
    transparent_cams = False
    cam_size = 0.05
    scenegraph_type = "complete"
    winsize = 1
    refid = 0
    # all_info cuda False 512 ['/tmp/gradio/8df9d5949578ec91fd98805367183ce574801453/vis_0_1.png', '/tmp/gradio/a26c13cba5c2675ffc9e8289d9bd5c20b0fae128/vis_0_0.png'] linear 300 3 False False True False 0.05 complete 1 0
    print('all_info', device, silent, image_size, filelist, schedule, niter, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize, refid)
    
    imgs = load_images(filelist, size=image_size, verbose=not silent) # image resize inside
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    
    torch.cuda.synchronize()
    t = [time.time()]
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    torch.cuda.synchronize()
    t.append(time.time())

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    torch.cuda.synchronize()
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
    torch.cuda.synchronize()
    t.append(time.time())

    print('test net inference time', t[1] - t[0], 'GO time', t[2] - t[1])

    pts_3d = scene.get_pts3d() # in the first cam sys
    rgbs = scene.imgs # list of [h, w, 3]
    c2w = scene.get_im_poses()
    for x in pts_3d:
        print(x.shape) # [h, w, 3]
    print('c2w', c2w.shape, c2w) # [n, 4, 4]
    all_pcd = torch.cat([pcd.reshape(-1, 3).detach().cuda() for pcd in pts_3d], dim = 0)
    all_pcd = c2w[0,:3,3] + all_pcd @ c2w[0,:3,:3].T
    all_rgb = torch.cat([torch.from_numpy(rgb.reshape(-1, 3)).cuda() for rgb in rgbs], dim = 0)

    return 
    # pcd_render(all_pcd, all_rgb, tgt = "./all.mp4", normalize = True)
    # outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
    #                                   clean_depth, transparent_cams, cam_size)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d/depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))
    exit(0)
    # return scene, outfile, imgs
    
def Rt(M, p):

    return M[:3,3] + p @ M[:3,:3].T

def inference_global_optimization(model, device, silent, img_tensors, first_view_c2w): # (model), cuda, False, 512, [...,...]
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    schedule = "linear"
    niter = 300
    min_conf_thr = 3
    as_pointcloud = True
    mask_sky = False
    clean_depth = False
    transparent_cams = False
    cam_size = 0.05
    scenegraph_type = "complete"
    winsize = 1
    refid = 0
    imgs = []
    for img_id, img in enumerate(img_tensors):
        print('img inference', img.shape, img_id)
        imgs.append(dict(img = img[None], true_shape=np.int32([img.shape[-2:]]), idx=img_id, instance=str(img_id)))
    # imgs = load_images(filelist, size=image_size, verbose=not silent) # image resize inside
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)

    t = [time.time()]
    torch.cuda.synchronize()
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
    torch.cuda.synchronize()
    t.append(time.time())

    torch.cuda.synchronize()
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
    torch.cuda.synchronize()
    t.append(time.time())

    print('test net inference time', t[1] - t[0], 'GO time', t[2] - t[1])
    pts_3d = scene.get_pts3d() # in the first cam sys
    conf = scene.get_conf()
    # rgbs = scene.imgs # list of [h, w, 3]
    # c2w = first_view_c2w
    # for x in pts_3d:
    #     print(x.shape) # [h, w, 3]
    # print('c2w', c2w.shape, c2w) # [n, 4, 4]
    # all_pcd = torch.cat([pcd.reshape(-1, 3).detach().cuda() for pcd in pts_3d], dim = 0)
    # all_pcd = c2w[:3,3] + all_pcd @ c2w[:3,:3].T

    output_pcd = []
    vis_pcd = []
    all_c2w = scene.get_im_poses()
    intrinsics = scene.get_intrinsics()
    # all_c2w = [torch.linalg.inv(w2c) for w2c in all_w2c]
    for pcd in pts_3d:
        pcd_original_shape = pcd.shape

        original_first_w2c = torch.linalg.inv(scene.get_im_poses()[0])
        pcd_c = Rt(original_first_w2c, pcd.reshape(-1, 3))
        
        output_pcd.append(pcd_c.reshape(*pcd_original_shape))

        # pcd_transformed = Rt(c2w, pcd_c)
        # vis_pcd.append(pcd_transformed)
    
    # vis_pcd = torch.stack(vis_pcd, dim = 0).reshape(-1, 3)
    # vis_rgb = torch.cat([torch.from_numpy(rgb.reshape(-1, 3)).cuda() for rgb in rgbs], dim = 0)
    # pcd_render(vis_rgb, vis_pcd, tgt = "./all.mp4", normalize = True)

    return output_pcd, all_c2w, intrinsics, conf, t[1] - t[0], t[2] - t[1]

    # outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
    #                                   clean_depth, transparent_cams, cam_size)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d/depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    if scenegraph_type == "swin":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size)
    recon_fun(["/home/zgtang/manifold_things/sample_img/vis_0_0.png", "/home/zgtang/manifold_things/sample_img/vis_0_1.png", "/home/zgtang/manifold_things/sample_img/vis_0_0.png", "/home/zgtang/manifold_things/sample_img/vis_0_1.png"])
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)

    recon_fun(inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                mask_sky, clean_depth, transparent_cams, cam_size,
                scenegraph_type, winsize, refid)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # dust3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent)
