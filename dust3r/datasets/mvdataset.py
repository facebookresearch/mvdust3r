import sys, os
import os.path as osp
import json
import itertools
from collections import deque
import imageio
from copy import deepcopy

import cv2
import numpy as np
import random
import h5py

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

if 'META_INTERNAL' in os.environ.keys() and os.environ['META_INTERNAL'] == "False":
    from dust3r.dummy_io import *
else:
    from meta_internal.io import *
    
class MVDataset(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, from_tar = False, random_order = False, random_render_order = False, debug = False, *args, ROOT, n_test=1000, num_render_views = 0, n_vis_test = 8, n_vis_train = 8, split_thres = 0.9, render_start = None, tb_name = None, ref_all = False, n_ref = 1, random_nv_nr = None, dps_name = 'dps.h5', n_all = None, single_id = None, reverse = False, **kwargs):
        self.ROOT = ROOT
        self.num_render_views = num_render_views
        self.from_tar = from_tar
        self.random_order = random_order
        self.random_render_order = random_render_order
        super().__init__(*args, **kwargs) # self.num_views, split set inside
        if "test" in dps_name:
            self.test = True
        else:
            self.test = False
        self.num_inference_views = self.num_views - self.num_render_views
        self.num_vis_test = n_vis_test
        self.num_vis_train = n_vis_train
        if render_start is None:
            render_start = self.num_inference_views
        self.render_start = render_start
        self.tb_name = tb_name if tb_name is not None else self.split
        self.ref_all = ref_all
        self.n_ref = n_ref
        if random_nv_nr is None:
            random_nv_nr = [[self.num_views, self.num_render_views]]
        self.random_nv_nr = random_nv_nr
        self.dps_name = dps_name
        self.single_id = single_id

        # load all scenes
        self.data_name = osp.basename(self.ROOT)
        self.h5f_path = g_pathmgr.get_local_path(osp.join(self.ROOT, self.dps_name))
        with h5py.File(self.h5f_path, 'r') as h5f:
            self.dps = [i for i in range(len(h5f['json_strs']))]
        
        if self.split != "all":
            split_ind = int(len(self.dps) * split_thres)
            test_id_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_test)]
            train_id_list = np.setdiff1d(np.arange(split_ind), test_id_list)
            train_test_id_list = [x + 1 for x in range(0, split_ind, split_ind // n_test)]
            vis_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_vis_test)][-n_vis_test:] + [train_test_id_list[x] for x in range(0, len(train_test_id_list), len(train_test_id_list) // n_vis_train)][-n_vis_train:]
            # vis_list = [x for x in range(split_ind, len(self.dps), (len(self.dps) - split_ind) // n_vis_test)][-n_vis_test:] + [train_test_id_list[x] for x in range(0, len(train_test_id_list), len(train_test_id_list) // n_vis_train)][-n_vis_train:]
            print('vis list', len(vis_list), vis_list)
        
        if self.split == "all":
            if n_all is not None:
                self.dps = [self.dps[int(id / n_all * len(self.dps))] for id in range(n_all)]
            pass
        elif self.split == "train":
            self.dps = [self.dps[x] for x in train_id_list]
        elif self.split == "train_test":
            self.dps = [self.dps[x] for x in train_test_id_list]
        elif self.split == "vis":
            self.dps = [self.dps[x] for x in vis_list]
        elif self.split == "test":
            self.dps = [self.dps[x] for x in test_id_list]
        if self.single_id is not None:
            self.dps = [self.dps[self.single_id]]
        if reverse:
            self.dps = list(reversed(self.dps))
    def __len__(self):
        
        print('len in dataset', len(self.dps), self.split)
        if self.ref_all:
            return len(self.dps) * self.num_inference_views
        return len(self.dps)

    def _get_views(self, idx, resolution, rng, from_tar = False, ref_view_id = None):
        
        random_nv_nr = random.choice(self.random_nv_nr)
        # self.num_views = random_nv_nr[0]
        # self.num_render_views = random_nv_nr[1]
        # self.num_inference_views = self.num_views - self.num_render_views
        if ref_view_id is None:
            ref_view_id = 0
        if self.ref_all:
            ref_view_id = idx % self.num_inference_views
            idx = idx // self.num_inference_views
        with h5py.File(self.h5f_path, 'r') as h5f:
            json_str = h5f['json_strs'][self.dps[idx]]
            data_dict = json.loads(json_str)
            C = data_dict['C'] if 'C' in data_dict.keys() else None
            if C is None:
                C_avg = np.array(0.).astype(np.float32)
            else:
                C = np.array(C)
                C = C[:self.num_inference_views,:self.num_inference_views]
                C_avg = C.mean()
                if C[0][0] > 0.9:
                    C_avg -= 1 / self.num_inference_views
                C_avg = np.array(C_avg).astype(np.float32)
            
            rgb_list = data_dict['rgb_list']
            depth_list = data_dict['depth_list']
            pose_raw_list, pose_list = [], []
            if "pose_raw_list" in data_dict.keys():
                pose_raw_list = data_dict['pose_raw_list']
            else:
                pose_list = data_dict['pose_list']
            intrinsic_raw, intrinsic_list = None, []
            if "intrinsic_raw" in data_dict.keys():
                intrinsic_raw = data_dict['intrinsic_raw']
            else:
                intrinsic_list = data_dict['intrinsic_list']

            if len(rgb_list) < self.num_views and 'nv' not in data_dict.keys(): # not implemented
                n_repeat = (self.num_views - 1) // len(rgb_list) + 1
                rgb_list = (rgb_list * n_repeat)[:self.num_views]
                depth_list = (depth_list * n_repeat)[:self.num_views]
                pose_raw_list = (pose_raw_list * n_repeat)[:self.num_views]
                pose_list = (pose_list * n_repeat)[:self.num_views]
                intrinsic_list = (intrinsic_list * n_repeat)[:self.num_views]
        
        num_tuple = len(rgb_list)
        if 'nv' in data_dict.keys():
            num_tuple = data_dict['nv']
            rgb_all = imageio.imread(g_pathmgr.get_local_path(rgb_list[0])).astype(np.uint8)
            depth_all = imageio.imread(g_pathmgr.get_local_path(depth_list[0])).astype(np.float32) / 1000
            rgb_all = np.split(rgb_all, num_tuple, axis=1)
            depth_all = np.split(depth_all, num_tuple, axis=1)
            if len(rgb_all) < self.num_views:
                n_repeat = (self.num_views - 1) // len(rgb_all) + 1
                pose_raw_list = (pose_raw_list * n_repeat)[:self.num_views]
                pose_list = (pose_list * n_repeat)[:self.num_views]
                intrinsic_list = (intrinsic_list * n_repeat)[:self.num_views]
                rgb_all = (rgb_all * n_repeat)[:self.num_views]
                depth_all = (depth_all * n_repeat)[:self.num_views]
        if self.random_render_order:
            # randomly choose num_render_views from num_views w/o replacement:
            render_set = np.random.choice(range(self.num_views), size = self.num_render_views, replace = False)
        else:
            render_set = [self.render_start + i for i in range(self.num_render_views)]
        
        inference_set = []
        for i in range(self.num_views):
            if i not in render_set:
                inference_set.append(i)
            if len(inference_set) == self.num_inference_views:
                break

        rgb_list, depth_list, pose_list, intrinsic_list = change_to_sr([rgb_list, depth_list, pose_list, intrinsic_list])
        views = []
        for i in inference_set + render_set:
            
            if 'nv' in data_dict.keys():
                rgb = rgb_all[i]
                depth = depth_all[i]
            else:
                rgb = imageio.imread(g_pathmgr.get_local_path(rgb_list[i])).astype(np.uint8) # / 256
                if depth_list[i].endswith('.npy'):
                    depth = np.load(g_pathmgr.get_local_path(depth_list[i]))
                else:
                    depth = imageio.imread(g_pathmgr.get_local_path(depth_list[i])).astype(np.float32) / 1000
            
            if intrinsic_raw is not None:
                intrinsic_ = np.array(intrinsic_raw).astype(np.float32)
                intrinsic = np.zeros((4,4)).astype(np.float32)
                intrinsic[:3,:3] = intrinsic_[:3,:3]
            else:
                intrinsic = np.loadtxt(g_pathmgr.get_local_path(intrinsic_list[i])).astype(np.float32)
            if len(pose_raw_list):
                camera_pose = np.array(pose_raw_list[i]).astype(np.float32)
            else:
                camera_pose = np.loadtxt(g_pathmgr.get_local_path(pose_list[i])).astype(np.float32)
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))

            rgb, depth, intrinsic = self._crop_resize_if_necessary(
                rgb, depth, intrinsic, resolution, rng=rng, info="[Empty]")
            if self.save_results:
                if i < len(rgb_list):
                    if 'gibson' in rgb_list[i]:
                        scene_name = rgb_list[i].split('/')[-2]
                    elif 'hm3d' in rgb_list[i]:
                        scene_name = rgb_list[i].split('/')[-3]
                    elif 'mp3d' in rgb_list[i]:
                        scene_name = rgb_list[i].split('/')[-2]
                    else:
                        scene_name = rgb_list[i].split('/')
                        for x in scene_name:
                            if "scene" in x:
                                scene_name = x
                                break
                label=f"dataName_{self.tb_name}_id_{str(idx).zfill(9)}_sceneName_{scene_name}_refId_{str(ref_view_id).zfill(3)}"
            else:
                label=f"{str(idx).zfill(9)}"
                
            views.append(dict(
                random_nv_nr=np.array(random_nv_nr),
                img=rgb,
                depthmap=depth,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsic,
                dataset=self.data_name,
                label=label,
                instance=str(idx),
                only_render = i in render_set,
                num_render_views = random_nv_nr[1],
                n_ref = self.n_ref,
                C_avg = C_avg,
            ))
        
        if ref_view_id != 0:
            views[0], views[ref_view_id] = deepcopy(views[ref_view_id]), deepcopy(views[0])

        if self.random_order:
            random.shuffle(views)
        else:
            views[0], views[1] = deepcopy(views[1]), deepcopy(views[0])
        
        if self.num_inference_views < 12:
            if len(views) > 3:
                views[1], views[3] = deepcopy(views[3]), deepcopy(views[1])
            
            if len(views) > 6:
                views[2], views[6] = deepcopy(views[6]), deepcopy(views[2])
        else:
            
            change_id = self.num_inference_views // 4 + 1
            views[1], views[change_id] = deepcopy(views[change_id]), deepcopy(views[1])
            change_id = (self.num_inference_views * 2) // 4 + 1
            views[2], views[change_id] = deepcopy(views[change_id]), deepcopy(views[2])
            change_id = (self.num_inference_views * 3) // 4 + 1
            views[3], views[change_id] = deepcopy(views[change_id]), deepcopy(views[3])

        views_inference, views_render = [], []
        for view in views:
            if view['only_render']:
                views_render.append(view)
            else:
                views_inference.append(view)
        views = views_inference + views_render
        assert len(views) == self.num_views
        
        return views

if __name__ == "__main__":
    pass
