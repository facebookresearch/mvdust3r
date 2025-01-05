import torch
import torch.nn as nn
import numpy as np
import imageio
import os

if 'META_INTERNAL' in os.environ.keys() and os.environ['META_INTERNAL'] == "False":
    from dust3r.dummy_io import *
else:
    from meta_internal.io import *

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dust3r.pcd_render import spiral_cam_gen, save_video_combined

from gsplat.rendering import rasterization
from gsplat.rendering import spherical_harmonics

class GaussianRenderer(nn.Module):
    def __init__(self, im_height = 224, im_width = 224, znear=0.01, zfar=100.0):
        super().__init__()
        self.im_height = int(im_height)
        self.im_width = int(im_width)
        self.znear = znear
        self.zfar = zfar

        self.register_buffer("bg_color", torch.ones((1, 3), dtype=torch.float32))

    def set_view_info(self, height=0, width=0, znear=0.01, zfar=100.0):
        self.im_height = int(height)
        self.im_width = int(width)
        self.znear = znear
        self.zfar = zfar

    def compute_proj(self, tanfovx, tanfovy):
        top = tanfovy * self.znear
        bottom = -top
        right = tanfovx * self.znear
        left = -right

        P = torch.zeros(4, 4)
        z_sign = 1.0

        P[0, 0] = 2.0 * self.znear / (right - left)
        P[1, 1] = 2.0 * self.znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        return P

    def compute_intrinsic(self, tanfovx, tanfovy):
        Ks = torch.eye(3)
        Ks[0, 0] = self.im_width / tanfovx / 2.0
        Ks[1, 1] = self.im_height / tanfovy / 2.0
        Ks[0, 2] = self.im_width / 2.0
        Ks[1, 2] = self.im_height / 2.0
        return Ks

    def calc_color_from_sh(self, pcds, c2ws, sh, sh_degree, debug = False): # pcds: [nc, N, 3], c2ws: [nc, 4, 4], sh: [nc, N, K, 3] -> colors: [nc, N, 3] -1~1
        
        # def spherical_harmonics (
        #     degrees_to_use: int,
        #     dirs: Tensor,  # [..., 3]
        #     coeffs: Tensor,  # [..., K, 3]
        #     masks: Optional[Tensor] = None,
        # ) -> Tensor:
        #     """Computes spherical harmonics.

        #     Args:
        #         degrees_to_use: The degree to be used.
        #         dirs: Directions. [..., 3]
        #         coeffs: Coefficients. [..., K, 3]
        #         masks: Optional boolen masks to skip some computation. [...,] Default: None.

        #     Returns:
        #         Spherical harmonics. [..., 3]
        #     """
        dirs = pcds - c2ws[:, :3, 3][..., None, :] # [nc, N, 3]
        colors = spherical_harmonics(sh_degree, dirs, sh)
        colors = colors + 0.5
        # colors = torch.clamp_min(colors, 0)
        if debug:
            import fbvscode
            fbvscode.set_trace()
        colors = colors * 2 - 1
        return colors

    def forward(self, w2cs, Ks, xyz, rgb, opacity, scale, rotation, eps2d=0.3, SH = False, debug = False): # we assume the input rgb should be -1~1 if it is not sh
        if rgb.shape[-1] != 3 or SH:
            sh_degree = rgb.shape[-1] // 3
            rgb = rgb.reshape(-1, sh_degree, 3)
        if rgb.ndim == 2: # in color space -1 ~ 1
            sh_degree = None
        else:
            sh_base = rgb.shape[1]
            sh_degree = int(np.sqrt(sh_base)) - 1

        # Set up rasterization configuration
        extr = w2cs
        n_cam = extr.shape[0]
        bg_color = self.bg_color.repeat(n_cam, 1).to(extr.device)
        if debug:
            import fbvscode
            fbvscode.set_trace()
        
        out_img, out_alpha, _ = rasterization(
            means=xyz,
            quats=rotation,
            scales=scale,
            opacities=opacity,
            colors=rgb,
            viewmats=extr,
            Ks=Ks,
            width=self.im_width,
            height=self.im_height,
            near_plane=self.znear,
            far_plane=self.zfar,
            backgrounds=bg_color,
            sh_degree=sh_degree,
            eps2d=eps2d,
        )
        if sh_degree is not None: # input is SH, return image is 0~1
            out_img = out_img * 2 - 1 # original: 4, 512, 512, 3
        
        # print('rgb in GaussianRenderer', rgb.shape, out_img.shape, out_img.max(), out_img.min(), sh_degree)
        out_alpha = out_alpha # original: 4, 512, 512, 1

        return {"rgb": out_img, "mask": out_alpha}

def gs_render(gts, preds, dp_id_gt, dp_id_pred, c2w_canonical, normalize = False, rot = True, gt_img = False, gt_pcd = False):

    # gt1, gt2s, pred1, pred2s = gts[0], gts[1:], preds[0], preds[1:]
    # gt_pts1, gt_pts2s, pr_pts1, pr_pts2s, c2ws = torch.load('/home/zgtang/others.pt')
    # c2ws = torch.stack([c2w[dp_id] for c2w in c2ws], 0).cuda()
    # import fbvscode
    # c2ws = torch.stack([gt1['camera_pose'][dp_id]] + [gt2['camera_pose'][dp_id] for gt2 in gt2s], 0).cuda() # single: [4,4]
    intrinsics = torch.stack([gt['camera_intrinsics'][dp_id_gt][:3,:3] for gt in gts]).cuda() # 3,3
    
    rot_gs = torch.stack([pred['rotation'][dp_id_pred] for pred in preds], 0).reshape(-1, 4)
    scale_gs = torch.stack([pred['scale'][dp_id_pred] for pred in preds], 0).reshape(-1, 3)
    opacity_gs = torch.stack([pred['opacity'][dp_id_pred] for pred in preds], 0).reshape(-1)
    if gt_pcd:
        pts3d = torch.cat([gt['pts3d'][dp_id_gt] for gt in gts], 0).reshape(-1, 3).cuda() # [224,224,3]
    else:
        pts3d = torch.cat([preds[0]['pts3d'][dp_id_pred]] + [pred['pts3d_in_other_view'][dp_id_pred] for pred in preds[1:]], 0).reshape(-1, 3).cuda() # [224,224,3]
    if gt_img:
        imgs = torch.stack([gt['img'][dp_id_gt] for gt in gts], 0).permute(0,2,3,1).reshape(-1, 3).cuda() # single: [3,224,224]
        rot_gs = torch.ones_like(rot_gs)
        scale_gs = torch.ones_like(scale_gs) * 1e-3
        opacity_gs = torch.ones_like(opacity_gs) * 0.5
    else:
        imgs = torch.stack([pred['rgb'][dp_id_pred] for pred in preds], 0).flatten(0, -2).cuda() # [nv, ]
        sh_base = imgs.shape[-1] // 3
        imgs = imgs.reshape(-1, sh_base, 3)
    # fbvscode.set_trace()
    
    # pts3d2 = torch.cat([gt_pts1[dp_id]] + [gt_pts2[dp_id] for gt_pts2 in gt_pts2s], 0).reshape(-1, 3).cuda()
    gs = GaussianRenderer()
    # def forward(self, w2cs, Ks, xyz, rgb, opacity, scale, rotation):
    pts3d_rate = 2.
    pcd_range = (torch.max(pts3d, dim = 0)[0] - torch.min(pts3d, dim = 0)[0]).max()
    pts3d = pts3d / pcd_range
    pts3d = pts3d - torch.mean(pts3d, dim = 0)
    pts3d *= pts3d_rate
    if not gt_img:
        scale_gs = scale_gs / pcd_range * pts3d_rate
        scale_range = [0.0001, 0.004]
        # scale_range = [0.0001, 0.02]
        scale_gs = torch.clamp(scale_gs, scale_range[0], scale_range[1])

    w2cs = spiral_cam_gen(imgs.device, 36)
    c2ws = torch.linalg.inv(w2cs)
    c2ws[:,:,:2] *= -1

    R_ = torch.Tensor([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]).to(c2ws)
    R = torch.eye(4).to(R_)
    R[:3,:3] = R_
    # c2ws2 = torch.linalg.inv(c2w_canonical) @ R @ c2ws
    # pts3d
    c2ws2 = R @ c2ws
    if not gt_pcd:
        c2w_canonical_inv = torch.linalg.inv(c2w_canonical)
        c2w_canonical_inv[:3,3] = 0
        c2ws2 = c2w_canonical_inv @ c2ws2

    intrinsics = intrinsics[0].repeat(c2ws.shape[0], 1, 1)
    # if gt_img:
    #     import fbvscode
    #     fbvscode.set_trace()
    res = gs(torch.linalg.inv(c2ws2), intrinsics, pts3d, imgs, opacity_gs, scale_gs, rot_gs, eps2d = 0.1)
    # res2 = gs(torch.linalg.inv(c2ws), intrinsics, pts3d2, imgs, torch.ones_like(pts3d[:,0]) * 0.5, torch.ones_like(pts3d) * 0.001, rot)
    rgb = res['rgb']
    video_frames = [((((rgb[i].detach().cpu() + 1) / 2)).float().numpy() * 256).astype(np.uint8) for i in range(len(rgb))]
    # save_video_combined([video_frames], "/home/zgtang/spiral/0.mp4")
    return video_frames


if __name__ == '__main__':
    # data_info = {
    #     "scene_name": "scene0000_00",
    #     "rgb_list": [
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/color/3701.jpg",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/color/4245.jpg",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/color/2998.jpg",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/color/3864.jpg"
    #     ],
    #     "depth_list": [
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/depth/3701.png",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/depth/4245.png",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/depth/2998.png",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/depth/3864.png"
    #     ],
    #     "pose_list": [
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/pose/3701.txt",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/pose/4245.txt",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/pose/2998.txt",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/pose/3864.txt"
    #     ],
    #     "intrinsic_list": [
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/intrinsic/intrinsic_depth.txt",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/intrinsic/intrinsic_depth.txt",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/intrinsic/intrinsic_depth.txt",
    #         "manifold://scannet_dataset/tree/scans/scene0000_00/frames/intrinsic/intrinsic_depth.txt"
    #     ]
    # }

    if False:
        dp_id = 0
        gt1, gt2s, pred1, pred2s = torch.load('/home/zgtang/gt_pred.pt')
        gt_pts1, gt_pts2s, pr_pts1, pr_pts2s, c2ws = torch.load('/home/zgtang/others.pt')
        c2ws = torch.stack([c2w[dp_id] for c2w in c2ws], 0).cuda()
        import fbvscode
        imgs = torch.stack([gt1['img'][dp_id]] + [gt2['img'][dp_id] for gt2 in gt2s], 0).permute(0,2,3,1).reshape(-1, 3).cuda() # single: [3,224,224]
        # c2ws = torch.stack([gt1['camera_pose'][dp_id]] + [gt2['camera_pose'][dp_id] for gt2 in gt2s], 0).cuda() # single: [4,4]
        intrinsics = torch.stack([gt1['camera_intrinsics'][dp_id][:3,:3]] + [gt2['camera_intrinsics'][dp_id][:3,:3] for gt2 in gt2s]).cuda() # 3,3
        # pts3d = torch.cat([gt1['pts3d'][dp_id]] + [gt2['pts3d'][dp_id] for gt2 in gt2s], 0).reshape(-1, 3).cuda() # [224,224,3]
        pts3d = torch.cat([pr_pts1[dp_id]] + [pr_pts2[dp_id] for pr_pts2 in pr_pts2s], 0).reshape(-1, 3).cuda() # [224,224,3]
        # pts3d2 = torch.cat([gt_pts1[dp_id]] + [gt_pts2[dp_id] for gt_pts2 in gt_pts2s], 0).reshape(-1, 3).cuda()
        gs = GaussianRenderer()
        # def forward(self, w2cs, Ks, xyz, rgb, opacity, scale, rotation):
        rot = torch.ones(pts3d.shape[0], 4).to(pts3d)
        res = gs(torch.linalg.inv(c2ws), intrinsics, pts3d, imgs, torch.ones_like(pts3d[:,0]) * 0.5, torch.ones_like(pts3d) * 0.001, rot)
        # res2 = gs(torch.linalg.inv(c2ws), intrinsics, pts3d2, imgs, torch.ones_like(pts3d[:,0]) * 0.5, torch.ones_like(pts3d) * 0.001, rot)
        rgb = res['rgb']
        import imageio
        for i in range(4):
            imageio.imwrite(f"./{i}_.png", (((rgb[i].detach().cpu() + 1) / 2)).float())
        fbvscode.set_trace()
    
    if True:
        dp_id = 0
        gt1, gt2s, pred1, pred2s = torch.load('/home/zgtang/gt_pred.pt')
        gt_pts1, gt_pts2s, pr_pts1, pr_pts2s, c2ws = torch.load('/home/zgtang/others.pt')
        c2ws = torch.stack([c2w[dp_id] for c2w in c2ws], 0).cuda()
        import fbvscode
        imgs = torch.stack([gt1['img'][dp_id]] + [gt2['img'][dp_id] for gt2 in gt2s], 0).permute(0,2,3,1).reshape(-1, 3).cuda() # single: [3,224,224]
        # c2ws = torch.stack([gt1['camera_pose'][dp_id]] + [gt2['camera_pose'][dp_id] for gt2 in gt2s], 0).cuda() # single: [4,4]
        intrinsics = torch.stack([gt1['camera_intrinsics'][dp_id][:3,:3]] + [gt2['camera_intrinsics'][dp_id][:3,:3] for gt2 in gt2s]).cuda() # 3,3
        pts3d = torch.cat([gt1['pts3d'][dp_id]] + [gt2['pts3d'][dp_id] for gt2 in gt2s], 0).reshape(-1, 3).cuda() # [224,224,3]
        # pts3d = torch.cat([pr_pts1[dp_id]] + [pr_pts2[dp_id] for pr_pts2 in pr_pts2s], 0).reshape(-1, 3).cuda() # [224,224,3]
        # pts3d2 = torch.cat([gt_pts1[dp_id]] + [gt_pts2[dp_id] for gt_pts2 in gt_pts2s], 0).reshape(-1, 3).cuda()
        gs = GaussianRenderer()
        # def forward(self, w2cs, Ks, xyz, rgb, opacity, scale, rotation):
        pcd_range = (torch.max(pts3d, dim = 0)[0] - torch.min(pts3d, dim = 0)[0]).max()
        pts3d = pts3d / pcd_range
        pts3d = pts3d - torch.mean(pts3d, dim = 0)
        pts3d *= 3
        rot = torch.ones(pts3d.shape[0], 4).to(pts3d)
        w2cs = spiral_cam_gen(c2ws.device, 20)
        c2ws = torch.linalg.inv(w2cs)
        c2ws[:,:,:2] *= -1

        R_ = torch.Tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ]).to(c2ws)
        R = torch.eye(4).to(R_)
        R[:3,:3] = R_
        c2ws2 = R @ c2ws

        intrinsics = intrinsics[0].repeat(c2ws.shape[0], 1, 1)
        res = gs(torch.linalg.inv(c2ws2), intrinsics, pts3d, imgs, torch.ones_like(pts3d[:,0]) * 0.5, torch.ones_like(pts3d) * 0.001, rot)
        # res2 = gs(torch.linalg.inv(c2ws), intrinsics, pts3d2, imgs, torch.ones_like(pts3d[:,0]) * 0.5, torch.ones_like(pts3d) * 0.001, rot)
        rgb = res['rgb']
        import imageio
        video_frames = [((((rgb[i].detach().cpu() + 1) / 2)).float().numpy() * 256).astype(np.uint8) for i in range(20)]
        save_video_combined([video_frames], "/home/zgtang/spiral/0.mp4")
        # for i in range(20):
        #     imageio.imwrite(f"/home/zgtang/spiral/{i}.png", (((rgb[i].detach().cpu() + 1) / 2)).float())
        fbvscode.set_trace()
