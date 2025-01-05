import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import cv2
import time

# Util function for loading point clouds
import numpy as np
from pytorch3d.structures import Pointclouds

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
try:
    from iopath.common.file_io import g_pathmgr
except:
    pass

try:
    from fblearner.flow.facebook import fbtypes
    if os.path.exists("/usr/local/fbprojects/ffmpeg-ref/ffmpeg/bin/ffmpeg"):
        os.environ[
            "IMAGEIO_FFMPEG_EXE"
        ] = "/usr/local/fbprojects/ffmpeg-ref/ffmpeg/bin/ffmpeg"
    else:
        ffmpeg_pkg = fbtypes.FBPACKAGE.new("ffmpeg-ref:prod")
        ffmpeg_path = os.path.join(ffmpeg_pkg.package_path, "ffmpeg/bin/ffmpeg")
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
except:
    pass


def move_manifold(src, tgt):

    tgt = tgt.replace("manifold://", "")
    os.system(f"manifold --prod-use-cython-client put --threads 2 {src} {tgt}")

    print(f"manifold --prod-use-cython-client put --threads 2 {src} {tgt}")

def move_manifold2(src, tgt):

    # tgt = tgt.replace("manifold://", "")
    g_pathmgr.copy_from_local(src, tgt, overwrite = True)

def save_image_manifold(img, tgt):

    temp_name = f"/tmp/output_{time.time()}.png"
    imageio.imwrite(temp_name, img)
    move_manifold2(temp_name, tgt)

def spiral_cam_gen(device, n = 36, dis = 2.7):

    c2ws = []
    for i in range(0, 360, 360 // n):
        R, T = look_at_view_transform(dis, 10, i)
        c2w = torch.eye(4).to(device)
        c2w[:3,:3] = R.to(device)
        c2w[:3,3] = T.to(device).squeeze()
        c2ws.append(c2w)
    return torch.stack(c2ws)

def pcd_render(pcd, rgb, tgt = None, normalize = False, rot = True, mask = None, debug = False):

    pcd = pcd.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    # print('pcd_render', tgt)
    # torch.save([pcd, rgb], f"/home/zgtang/manifold_things/temp/pcd_rgb2.pt")

    # pcd[:,1] = pcd[:,1] * -1

    device = pcd.device
    if torch.min(rgb) < -0.5:
        rgb = (rgb + 1) / 2.0
    
    if normalize: 
        pcd_range = (torch.max(pcd, dim = 0)[0] - torch.min(pcd, dim = 0)[0]).max()
        pcd = pcd / pcd_range
        pcd = pcd - torch.mean(pcd, dim = 0)
        pcd *= 1.7

    R = torch.Tensor([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]).to(pcd)

    # multiply R
    if rot:
        pcd = torch.matmul(pcd, R)
    
    if mask is None:
        mask = torch.ones((pcd.shape[0], )).bool().to(pcd.device)
    else:
        mask = mask.reshape(-1)
    point_cloud = Pointclouds(points=[pcd[mask]], features=[rgb[mask]])
    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        radius = 0.003,
        points_per_pixel = 10,
        bin_size = None, # this drastically makes rendering faster, but sometimes cause overflow (then incomplete rendering). If you need complete rendering, set bin_size = 0.
    )

    images = []
    for i in range(0, 360, 10):
        R, T = look_at_view_transform(200, 30, i)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
        if debug:
            print('pcd rendering', i)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            # compositor=AlphaCompositor(background_color=(1, 1, 1))
            compositor=NormWeightedCompositor(background_color=(1, 1, 1))
        )
        with torch.no_grad():
            image = renderer(point_cloud)[0]
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        images.append(image)

    if tgt is None:
        return images

    file_name = f"output_{time.time()}.mp4"
    output_video = f'/tmp/{file_name}'

    # height, width, layers = 512, 512, 3

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

    # for image in images:
    #     video.write(image)

    # video.release()

    writer = imageio.get_writer(output_video, fps=20)
    
    for image in images:
        writer.append_data(image)

    writer.close()
    print(f'Video saved as {output_video}')

    if tgt[:11] == "manifold://":
        move_manifold2(output_video, tgt)
    else:
        os.system(f"mv {output_video} {tgt}")

def save_video_combined(images_list, tgt):

    images = []
    for i in range(len(images_list[0])):
        img_row = []
        for images_list_j in images_list:
            img_row.append(images_list_j[i])
        img_row = np.concatenate(img_row, axis = 1) # [h, w (cat here), 3]
        images.append(img_row)

    file_name = f"output_{time.time()}.mp4"
    output_video = f'/tmp/{file_name}'

    writer = imageio.get_writer(output_video, fps=5)
    
    for image in images:
        writer.append_data(image)

    writer.close()
    print(f'Video saved as {output_video}')

    if tgt[:11] == "manifold://":
        move_manifold2(output_video, tgt)
    else:
        os.system(f"mv {output_video} {tgt}")



if __name__ == '__main__': 
    
    tgt = f"manifold://ondevice_ai_writedata/tree/zgtang/dust3r/logs/torchx-dust3r_train-test4/videos/temp5.mp4"
    pcd_render_test(tgt, normalize = True)
