
import sys, os
from copy import deepcopy

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

if 'META_INTERNAL' in os.environ.keys() and os.environ['META_INTERNAL'] == "False":
    generate_html = None
    from dust3r.dummy_io import *
else:
    from meta_internal.io import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--div", default=1, type=int)
parser.add_argument("--node-no", default=0, type=int)
parser.add_argument("--hardness", type=str)
parser.add_argument("--n-v", type = int)
parser.add_argument("--n-render", type = int)
parser.add_argument("--data-type", type = str)
parser.add_argument("--split", type = str, default = "all")
parser.add_argument("--render-overlap", type = float, default = 0.95)
parser.add_argument("--n-tuple-per-scene", type = int, default = 1000)
args = parser.parse_args()
print('args', args)
node_no = args.node_no

import torch.distributed as dist

def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

init_distributed()
rank = get_rank()


cuda_id = int(rank) % 8
device = f"cuda:{cuda_id}"
print('cuda id', cuda_id)

import PIL
import numpy as np
import torch
import glob, sys
import json
import imageio
import cv2

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

train_name_list_path = g_pathmgr.get_local_path(global_variable.train_name_list_path)
train_name_list = json.load(open(global_variable.train_name_list_path, 'r'))

n_v = args.n_v
n_render = args.n_render
# n_v = 4
# n_render = 1
# data_type = "scannet"
data_type = args.data_type
hardness = args.hardness
split = args.split

n_tuple_per_scene = args.n_tuple_per_scene
n_try = 1000
target_n = 100
n_try_scene = 100000

render_range = [
    [0, 4 * (i + 1)]
    for i in range(6)
]

cover_hardness = {
    'hard': [0.1, 0.4],
    'easy': [0.3, 0.7],
    'easier': [0.3, 1.0],
}
if "_" in hardness:
    mi, ma = hardness.split('_')
    cover_hardness[hardness] = [float(mi), float(ma)]

n_inference = n_v - n_render


# dataset_name = "scannet_large_easy_12"
# dataset_name = "scannet_large_easier_12"
# dataset_name = "scannet_large_12"
# dataset_name = f"scannetpp_large_easier_{n_v}"
# dataset_name = f"scannetpp_large_easy_{n_v}"
dataset_name = f"{data_type}_{hardness}_{n_v}_seq_{split}"

if data_type == "scannet":
    nn_shrink_rate = 3
elif data_type == "scannetpp":
    nn_shrink_rate = 9

def compare(a, b):
    a = int(os.path.basename(a)[:-4])
    b = int(os.path.basename(b)[:-4])
    return a < b

def key(a):
    return int(os.path.basename(a)[:-4])

def min_dis(A, B):

    A = torch.from_numpy(A).reshape(-1, 3).to(device)
    B = torch.from_numpy(B).reshape(-1, 3).to(device)
    from pytorch3d.ops import knn_points
    dis, _, _ = knn_points(B[None], A[None]) # B querying in A, dis: [1, B.shape[0], 1]
    return dis[0,:,0]

def cover(pc1_, pc2_): # querying pc2 in pc1
    import numpy as np
    pc1 = pc1_.reshape(-1, 3)
    pc2 = pc2_.reshape(-1, 3)
    
    distances = min_dis(pc1, pc2)
    
    thres = 0.015 * nn_shrink_rate
    return distances[(distances > 0) * (distances < thres)].shape[0] / distances.shape[0]

def get_score(pc1, pc2):
    return (cover(pc1, pc2) + cover(pc2, pc1)) / 2

def extract_valid_frames(valid_frames_ss):
    valid_frames_name = []
    for x in valid_frames_ss:
        x = x.split(' ')
        if len(x) <= 1:
            break
        if int(x[2]) != 0:
            print('get bad frame', x)
            # exit(0)
        valid_frames_name.append(int(x[1]))
    return valid_frames_name

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def qt2w2c(q, t) -> np.ndarray:
    R = qvec2rotmat(q)
    world2cam = np.eye(4)
    world2cam[:3, :3] = R
    world2cam[:3, 3] = t
    return world2cam

def extract_image_txt(lines):
    c2ws = []
    frame_names = []
    for x in lines:
        if ".jpg" in x:
            x = x.split(' ')
            q = [float(y) for y in x[1:5]]
            t = [float(y) for y in x[5:8]]
            frame = int(x[9].replace('frame_', '').replace('.jpg', ''))
            c2ws.append(np.linalg.inv(qt2w2c(q, t)))
            frame_names.append(frame)
            # print('extract', x, q, t, frame)
    
    return c2ws, frame_names

def extract_K(lines):
    ss = lines[3].split(' ')
    K = np.eye(3)
    K[0,0] = float(ss[4])
    K[1,1] = float(ss[5])
    K[0,2] = float(ss[6])
    K[1,2] = float(ss[7])
    return K

def get_scene_list():
    if data_type == "scannet":
        scan_list = glob.glob("/vol22/zt15/scannet/scans/*")
        scan_list.sort()
        # scan_list = np.load(g_pathmgr.get_local_path(global_variable.scannet_scan_list))
    elif data_type == "scannetpp":
        scan_list = np.load(g_pathmgr.get_local_path(global_variable.scannetpp_scan_list))
        scan_list = [os.path.join(global_variable.scannetpp_data_dir + "/data", x.split('/')[-1]) for x in scan_list]
    return scan_list

def filter_test(scene_list):
    scene_list_new = []
    for x in scene_list:
        scene_name = os.path.basename(x)
        if scene_name not in train_name_list:
            scene_list_new.append(x)
    return scene_list_new

def main():
    
    scene_list = get_scene_list()
    if args.split == "test":
        scene_list = filter_test(scene_list)
    # exit(0)
    tuple_done = 0
    for scene_id, scene_name in enumerate(scene_list[::1]):
        if scene_id % args.div != cuda_id + args.node_no * 8:
            continue
        print('trying', scene_id, cuda_id, cuda_id + args.node_no * 8, scene_name)
        skip_scene = False
        
        
        target_scene_name = f"{global_variable.metadata_dir}/{dataset_name}/{os.path.basename(scene_name)}"
        last_name = f"{target_scene_name}/{str(n_tuple_per_scene - 1).zfill(6)}_extra.pt"
        if g_pathmgr.exists(last_name):
            print('exist', last_name)
            continue
        print('doing', scene_id, cuda_id, cuda_id + args.node_no * 8, scene_name)
        
        g_pathmgr.mkdirs(target_scene_name)
        
        print('scene_name', scene_name)
        pose_raw_list = []
        pose_list = []
        if data_type == "scannet":
            valid_frame_name = os.path.join(scene_name, "valid_frames.txt")
            if g_pathmgr.exists(valid_frame_name):
                with g_pathmgr.open(valid_frame_name, 'r') as f:
                    valid_frames_ss = f.readlines()
                valid_frames_name = extract_valid_frames(valid_frames_ss)
                rgb_suff = "jpg"
                if not g_pathmgr.exists(os.path.join(scene_name, "frames", "color", f"{valid_frames_name[0]}.{rgb_suff}")):
                    print('not exist', os.path.join(scene_name, "frames", "color", f"{valid_frames_name[0]}.{rgb_suff}"))
                    rgb_suff = "png"
            else:
                valid_frames_name = glob.glob(f"{scene_name}/frames/color/*")
                valid_frames_name.sort()
                rgb_suff = valid_frames_name[0].split('.')[-1]
                valid_frames_name = [os.path.basename(x).split('.')[0] for x in valid_frames_name]
            rgb_list = [os.path.join(scene_name, "frames", "color", f"{valid_frame}.{rgb_suff}") for valid_frame in valid_frames_name]
            depth_list = [os.path.join(scene_name, "frames", "depth", f"{valid_frame}.png") for valid_frame in valid_frames_name]
            pose_list = [os.path.join(scene_name, "frames", "pose", f"{valid_frame}.txt") for valid_frame in valid_frames_name]
            intrinsic_depth = np.loadtxt(g_pathmgr.get_local_path(os.path.join(scene_name, "frames", "intrinsic", "intrinsic_depth.txt")))
            intrinsic_depth_original = deepcopy(intrinsic_depth)

        elif data_type == "scannetpp":
            poses, frame_names = extract_image_txt(open(g_pathmgr.get_local_path(os.path.join(scene_name, "iphone", "colmap", "images.txt")), "r").readlines())
            intrinsic_depth = extract_K(open(g_pathmgr.get_local_path(os.path.join(scene_name, "iphone", "colmap", "cameras.txt")), "r").readlines())
            intrinsic_depth_original = deepcopy(intrinsic_depth)
            intrinsic_depth[:2] *= 1 / nn_shrink_rate
            rgb_list = []
            depth_list = []
            pose_raw_list = [x for x in poses]
            depth_dir = os.path.join(f"{global_variable.scannetpp_dir}/render", scene_name.split('/')[-1])
            for frame in frame_names:
                rgb_name   = os.path.join(scene_name, "iphone", "rgb", f"frame_{str(frame).zfill(6)}.jpg")
                depth_name = os.path.join(depth_dir, "iphone", "render_depth", f"frame_{str(frame).zfill(6)}.png")
                rgb_list.append(rgb_name)
                depth_list.append(depth_name)

        # step = max(len(rgb_list) // target_n, 1)
        id_list = [int(i / target_n * len(rgb_list)) for i in range(target_n)]
        rgb_list = [rgb_list[id] for id in id_list]
        depth_list = [depth_list[id] for id in id_list]
        if pose_list:
            pose_list = [pose_list[id] for id in id_list]
        if pose_raw_list:
            pose_raw_list = [pose_raw_list[id] for id in id_list]
        
        print('loading:', len(rgb_list))
        # rgb_preload = []
        depth_preload = []
        pose_preload = []
        valid_set = []
        for id in range(len(rgb_list)):
            try:
                rgb = imageio.imread(g_pathmgr.get_local_path(rgb_list[id])).astype(np.float32) / 255
                depth = imageio.imread(g_pathmgr.get_local_path(depth_list[id])).astype(np.float32) / 1000
                if len(pose_list) > 0:
                    pose = np.loadtxt(g_pathmgr.get_local_path(pose_list[id]))
                else:
                    pose = pose_raw_list[id]
                # rgb_preload.append(rgb)
                depth_preload.append(depth)
                pose_preload.append(pose)
                valid_set.append(id)
            except:
                pass
            print('loading', id)
        if len(depth_preload) < len(rgb_list) * 0.7:
            continue
        rgb_list = [rgb_list[x] for x in valid_set]
        depth_list = [depth_list[x] for x in valid_set]
        if len(pose_list):
            pose_list = [pose_list[x] for x in valid_set]
        if len(pose_raw_list):
            pose_raw_list = [pose_raw_list[x] for x in valid_set]
        print('load Done')
        for cnt in range(n_tuple_per_scene):

            if skip_scene:
                break
            
            cnt_all_test = 0
            current_name = f"{target_scene_name}/{str(cnt).zfill(6)}_extra.pt"
            if g_pathmgr.exists(current_name):
                print('secEx', current_name)
                continue
            # generate n_v views randomly from rgb_list and depth_list, do not replace.

            id_list = [None for i in range(n_v)]
            focal = (intrinsic_depth[0,0] + intrinsic_depth[1, 1]) / 2
            pcd_all = []
            C = np.zeros((n_v, n_v))
            i = 0
            rgbs = []
            cover_mask_all = [None for i in range(n_render)]
            start_id = np.random.randint(target_n - n_v)
            while i < n_v:
                            
                if skip_scene:
                    break

                try_cnt = 0

                while 1:

                    try_cnt += 1
                    id_list[i] = start_id + i # np.random.choice(len(depth_list))
                    cnt_all_test += 1

                    if try_cnt > n_try:
                        print('failed')
                        try_cnt = 0
                        i = -1
                        rgbs = []
                        pcd_all = []
                        C = np.zeros((n_v, n_v))
                        id_list = [None for i in range(n_v)]
                        break
                    
                    if id_list[i] in id_list[:i]:
                        continue

                    print('test', 'scene_id', scene_id, 'cnt', cnt, 'cnt_all_test', cnt_all_test, 'i', i, 'id_list[i]', id_list[i], 'try_cnt', try_cnt) # test 57 608 206 9 184 130
                    if cnt_all_test > n_try_scene:
                        print('failed scene')
                        skip_scene = True
                        break

                    import time
                    t = time.time()
                    id = id_list[i]
                    # rgb = rgb_preload[id]
                    depth = depth_preload[id]
                    depth = cv2.resize(depth, (depth.shape[1] // nn_shrink_rate, depth.shape[0] // nn_shrink_rate))
                    pose = pose_preload[id]
                    print('load time', time.time() - t)
                    t = time.time()
                    pcd, valid_mask = depthmap_to_absolute_camera_coordinates(depth, intrinsic_depth, pose) # pcd (480, 640, 3) (480, 640) 0.9917024739583333
                    # pcd = pcd[::nn_shrink_rate,::nn_shrink_rate]
                    # valid_mask = valid_mask[::nn_shrink_rate,::nn_shrink_rate]
                    pcd_valid = pcd[valid_mask]
                    print('depth calc time', time.time() - t)
                    # print('pcd', pcd.shape, valid_mask.shape, valid_mask.mean()) # pcd (480, 640, 3) (480, 640) 0.9874479166666666
                    if i < n_inference:
                        score_list = []
                        t = time.time()
                        for j in range(i):
                            score_list.append(get_score(pcd_all[j], pcd_valid))
                        print('nn time', time.time() - t)
                        print(score_list)

                        # if len(score_list) == 0 or (np.max(score_list) > 0.3 and np.max(score_list) < 0.7):
                        # if len(score_list) == 0 or (np.max(score_list) > 0.3 and np.max(score_list) < 1.0):
                        if 1:
                            pcd_all.append(pcd_valid)
                            try_cnt = 0
                            break
                    else:
                        i_render = i - n_inference
                        score_list = []
                        t = time.time()
                        pcd_all_combined = np.concatenate(pcd_all[render_range[i_render][0] : render_range[i_render][1]], 0)
                        # def cover(pc1_, pc2_): # querying pc2 in pc1
                        score = cover(pcd_all_combined, pcd_valid)
                        if score > args.render_overlap:
                            # cover_mask = cover(pcd_all_combined, pcd_original, return_mask = True)
                            # cover_mask = cover_mask.reshape(-1).cpu() * torch.from_numpy(valid_mask_original.reshape(-1)).float()
                            # cover_mask = cover_mask.reshape(depth.shape[0], depth.shape[1])
                            # cover_mask_all[i_render] = cover_mask
                            pcd_all.append(pcd_valid)
                            try_cnt = 0
                            break
                i += 1
                
                # rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
                # np.save(f"{target_scene_name}/{str(cnt).zfill(6)}_rgb_{i}.npy", rgb)
                # np.save(f"{target_scene_name}/{str(cnt).zfill(6)}_pcd_{i}.npy", pcd)
                # np.save(f"{target_scene_name}/{str(cnt).zfill(6)}_valid_{i}.npy", valid_mask)
                # np.save(f"{target_scene_name}/{str(cnt).zfill(6)}_focal_{i}.npy", focal)
                # np.save(f"{target_scene_name}/{str(cnt).zfill(6)}_pose_{i}.npy", pose)

                # rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
                # rgb = (rgb * 255).astype(np.uint8)
                # rgbs.append(rgb)
            
            tuple_done += 1
            # rgbs = np.concatenate(rgbs, axis=1)
            # imageio.imwrite(f"{target_scene_name}/{str(cnt).zfill(6)}_rgb.png", rgbs)

            if skip_scene:
                break

            C_avg = []
            for i in range(n_v):
                for j in range(n_v):
                    if i != j:
                        C[i,j] = get_score(pcd_all[i], pcd_all[j])
                        C_avg.append(C[i,j])
                    # print(i,j, C[i,j])
            
            rgb_list_ = [rgb_list[id] for id in id_list]
            depth_list_ = [depth_list[id] for id in id_list]

            if len(pose_list):
                pose_list_ = [pose_list[id] for id in id_list] # from here tomorrow
                pose_info = {"pose_list": pose_list_}
            else:
                pose_raw_list_ = [pose_raw_list[id].tolist() for id in id_list]
                pose_info = {"pose_raw_list": pose_raw_list_}

            extra_info = {
                'C': C.tolist(),
                'C_avg': np.mean(np.array(C_avg)).item(),
                'id_list': id_list,
                'rgb_list': rgb_list_,
                'depth_list': depth_list_,
                'intrinsic_raw': intrinsic_depth_original.tolist(),
                **pose_info,
                }
            print('extra_info', scene_id, scene_name, extra_info)
            torch.save(extra_info, f"{target_scene_name}/{str(cnt).zfill(6)}_extra.pt")
            # with open(f"{target_scene_name}/{str(cnt).zfill(6)}_extra.json", "w") as f:
            #     json.dump(extra_info, f, indent=4)
            
            print('C')
            print(C)
            print('tuple done', tuple_done, cuda_id, id_list)


main()
