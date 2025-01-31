# Copyright (C) 2025-present Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC BY-NC 4.0 (non-commercial use only).

#!/usr/bin/env python3
import argparse
import copy
import functools
import math
import os
import tempfile
from copy import deepcopy

import gradio
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

inf = np.inf

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from meta_internal.io import *

    os.environ["meta_internal"] = "True"
except:
    from dust3r.dummy_io import *

    os.environ["meta_internal"] = "False"

import matplotlib.pyplot as pl
from dust3r.inference import inference, inference_mv
from dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from dust3r.model import AsymmetricCroCo3DStereoMultiView
from dust3r.utils.device import to_numpy

from dust3r.utils.image import load_images, rgb
from dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, OPENGL, pts3d_to_trimesh

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=224, help="image size (note, we do not train and test on other resolutions yet, this should not be changed)")
    parser.add_argument("--server_port", type=int, help="will start gradio app on this port (if available).",
                        default=7860)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["MVD", "MVDp"])
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


def get_3D_model_from_scene(outdir, silent, output, min_conf_thr=3, as_pointcloud=False, transparent_cams=False, cam_size=0.05, only_model=False):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """

    with torch.no_grad():
        
        _, h, w = output['pred1']['rgb'].shape[0:3] # [1, H, W, 3]
        rgbimg = [output['pred1']['rgb'][0]] + [x['rgb'][0] for x in output['pred2s']]
        for i in range(len(rgbimg)):
            rgbimg[i] = (rgbimg[i] + 1) / 2
        pts3d = [output['pred1']['pts3d'][0]] + [x['pts3d_in_other_view'][0] for x in output['pred2s']]
        conf = torch.stack([output['pred1']['conf'][0]] + [x['conf'][0] for x in output['pred2s']], 0) # [N, H, W]
        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
        msk = conf >= conf_thres
        
        # calculate focus:

        conf_first = conf[0].reshape(-1) # [bs, H * W]
        conf_sorted = conf_first.sort()[0] # [bs, h * w]
        conf_thres = conf_sorted[int(conf_first.shape[0] * 0.03)]
        valid_first = (conf_first >= conf_thres) # & valids[0].reshape(bs, -1)
        valid_first = valid_first.reshape(h, w)

        focals = estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda()).cpu().item()

        intrinsics = torch.eye(3,)
        intrinsics[0, 0] = focals
        intrinsics[1, 1] = focals
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        intrinsics = intrinsics.cuda()

        focals = torch.Tensor([focals]).reshape(1,).repeat(len(rgbimg))

        
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda() # [H, W, 2]
        
        c2ws = []
        for (pr_pt, valid) in zip(pts3d, msk):
            c2ws_i = calibrate_camera_pnpransac(pr_pt.cuda().flatten(0,1)[None], pixel_coords.flatten(0,1)[None], valid.cuda().flatten(0,1)[None], intrinsics[None])
            c2ws.append(c2ws_i[0])

        cams2world = torch.stack(c2ws, dim=0).cpu() # [N, 4, 4]
        focals = to_numpy(focals)

        pts3d = to_numpy(pts3d)
        msk = to_numpy(msk)

    glb_file = _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)
    conf = to_numpy([x[0] for x in conf.split(1, dim=0)])
    rgbimg = to_numpy(rgbimg)
    if only_model:
        return glb_file
    return glb_file, rgbimg, conf


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, min_conf_thr,
                            as_pointcloud, transparent_cams, cam_size, n_frame):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent, n_frame = n_frame)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    for img in imgs:
        img['true_shape'] = torch.from_numpy(img['true_shape']).long()

    if len(imgs) < 12:
        if len(imgs) > 3:
            imgs[1], imgs[3] = deepcopy(imgs[3]), deepcopy(imgs[1])
        if len(imgs) > 6:
            imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])
    else:
        change_id = len(imgs) // 4 + 1
        imgs[1], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[1])
        change_id = (len(imgs) * 2) // 4 + 1
        imgs[2], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[2])
        change_id = (len(imgs) * 3) // 4 + 1
        imgs[3], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[3])
    
    output = inference_mv(imgs, model, device, verbose=not silent)
    input('press enter to continue')

    # print(output['pred1']['rgb'].shape, imgs[0]['img'].shape, 'aha')
    output['pred1']['rgb'] = imgs[0]['img'].permute(0,2,3,1)
    for x, img in zip(output['pred2s'], imgs[1:]):
        x['rgb'] = img['img'].permute(0,2,3,1)
    
    outfile, rgbimg, confs = get_3D_model_from_scene(outdir, silent, output, min_conf_thr, as_pointcloud, transparent_cams, cam_size)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    # rgbimg = scene.imgs
    # depths = to_numpy(scene.get_depthmaps())
    # confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    # depths_max = max([d.max() for d in depths])
    # depths = [d/depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]
    
    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        # imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return output, outfile, imgs


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
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent, only_model = True)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="MV-DUSt3R+ Demo", theme="default") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">MV-DUSt3R+ Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            
            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="confidence threshold (%)", value=3.0, minimum=0.0, maximum=20, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="camera size", value=0.05, minimum=0.001, maximum=0.5, step=0.001)
                
                n_frame = gradio.Slider(label="No. of video frames", value=10, minimum=4, maximum=100, step=1)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,confidence', columns=2, height="100%")

            # events
            
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, min_conf_thr, as_pointcloud,
                                  transparent_cams, cam_size, n_frame],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams, cam_size],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams, cam_size],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams, cam_size],
                                 outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams, cam_size],
                                    outputs=outmodel)
    demo.launch(share=True, server_name='127.0.0.1', server_port=args.server_port)

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

    weights_path = args.weights
    if args.model_name is None:
        if "MVDp" in args.weights:
            args.model_name = "MVDp"
        elif "MVD" in args.weights:
            args.model_name = "MVD"
        else:
            raise ValueError("model name not found in weights path")

    if args.model_name == "MVD":
        model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})
        model.to(args.device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(get_local_path(weights_path)).to(args.device)
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)
    elif args.model_name == "MVDp":
        model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True}, m_ref_flag=True, n_ref = 4)
        model.to(args.device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(get_local_path(weights_path)).to(args.device)
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)

    else:
        raise ValueError(f"{args.model_name} is not supported")


    # dust3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent)
