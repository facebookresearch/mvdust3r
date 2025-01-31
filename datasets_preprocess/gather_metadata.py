import glob
import os, sys
import json
import numpy as np
import torch
import h5py

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data-name")
parser.add_argument("--data-dir")
parser.add_argument("--tgt-dir")
# parser.add_argument("--node-no", default=0, type=int)
args = parser.parse_args()

if args.data_dir is None:
    args.data_dir = f"/home/zgtang/{args.data_name}"
else:
    args.data_name = os.path.basename(args.data_dir)

if args.tgt_dir is None:
    args.tgt_dir = args.data_dir

def extract_scene_name(x, data_name_):
    if "_meta" == data_name_[-5:]:
        data_name = data_name_[:-5]
    else:
        data_name = data_name_
    # print('extract', x, data_name)
    if "scannetpp" in x:
        return 'scannetpp'
    elif "scannet" in x:
        xx = x.split('/')
        for x_ in xx:
            if 'scene' in x_:
                return x_
    elif "habitat_sim" in x:
        if "gibson" in x:
            return 'gibson'
        if "mp3d" in x:
            return "mp3d"
        xx = x.split('/')
        for x_ in xx:
            if x_[5] == "-":
                return x_
    raise NotImplementedError
    
# /home/zgtang/misc/train_name_list.json
train_name_list = json.load(open("./data/train_name_list.json", 'r'))

def split_dps(x, dir_name):

    # makedirs
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data_train = []
    data_test = []
    for (d_id, d) in enumerate(x):
        rgb_path = d['rgb_list'][0]
        data_name = extract_scene_name(rgb_path, args.data_name)
        if data_name in train_name_list:
            data_train.append(d)
        else:
            data_test.append(d)
    
    print('train', len(data_train), 'test', len(data_test))
    if len(data_train) <= 2000:
        json.dump(data_train[::], open(f"{dir_name}/dps_train_sample.json", 'w'), indent=4)
        json.dump( data_test[::], open(f"{dir_name}/dps_test_sample.json", 'w'), indent=4)
    else:
        json.dump(data_train[::1000], open(f"{dir_name}/dps_train_sample.json", 'w'), indent=4)
        json.dump( data_test[::1000], open(f"{dir_name}/dps_test_sample.json", 'w'), indent=4)

    if len(data_train):
        print('dumping train')
        json_strs = [json.dumps(x) for x in data_train]
        ma_len = max(map(len, json_strs))
        json_strs = np.array(json_strs, dtype='S'+str(ma_len))
        with h5py.File(f"{dir_name}/dps_train.h5", 'w') as f:
            f.create_dataset('json_strs', data=json_strs, compression='gzip')
    
    if len(data_test):
        print('dumping test')
        json_strs = [json.dumps(x) for x in data_test]
        ma_len = max(map(len, json_strs))
        json_strs = np.array(json_strs, dtype='S'+str(ma_len))
        with h5py.File(f"{dir_name}/dps_test.h5", 'w') as f:
            f.create_dataset('json_strs', data=json_strs, compression='gzip')


def tuple_n_general_new(dir_name, tgt_name):
    
    if os.path.exists(f"{dir_name}/dps.h5"):
        x = h5py.File(f"{dir_name}/dps.h5", 'r')
        json_strs = x['json_strs']
        for x in json_strs[::1000]:
            print(json.loads(x)['rgb_list'])
        input()
        x = [json.loads(x) for x in json_strs]
        split_dps(x, dir_name)
        return
    if os.path.exists(f"{dir_name}/dps.json"):
        x = json.load(open(f"{dir_name}/dps.json", 'r'))
        split_dps(x, dir_name)
        return
    dps = []
    # metadata_list = glob.glob("/home/zgtang/scannet_large_easy/*/*extra.pt")
    pt_name = f"{dir_name}/*/*extra.pt"
    metadata_list = []
    for i in range(6):
        # print(pt_name)
        metadata_list = metadata_list + glob.glob(pt_name)
        pt_name = pt_name.replace("/*extra.pt", "/*/*extra.pt")
    sorted(metadata_list)
    print('tuple in sum', len(metadata_list))
    
    scene_cnt = {}
    for x_id, x in enumerate(metadata_list[::100]):
        dp = {}
        dp['scene_name'] = x.split('/')[-2]
        if dp['scene_name'] not in scene_cnt.keys():
            scene_cnt[dp['scene_name']] = 1
        else:
            scene_cnt[dp['scene_name']] += 1
    print('scene in sum', len(list(scene_cnt.keys())))
        
    scene_cnt = {}
    cnt_failed = 0
    for x_id, x_ in enumerate(metadata_list[::]):
        try:
            x = torch.load(x_)
            # print(x_)
            dp = {}
            dp['scene_name'] = x_.split('/')[-2]
            if dp['scene_name'] not in scene_cnt.keys():
                scene_cnt[dp['scene_name']] = 1
            else:
                scene_cnt[dp['scene_name']] += 1
            
            # print(dp['scene_name'], x)
            # print(x['rgb_list'], x['depth_list'], x['pose_list'])
            # print(x['intrinsic_list'])
            dp['rgb_list'] = x['rgb_list']
            dp['depth_list'] = x['depth_list']
            if 'pose_raw_list' in x.keys():
                dp['pose_raw_list'] = x['pose_raw_list']
            else:
                dp['pose_list'] = x['pose_list']
            if "nv" in x.keys():
                dp['nv'] = x['nv']
            if "intrinsic_raw" in x.keys():
                dp['intrinsic_raw'] = x['intrinsic_raw']
            else:
                dp['intrinsic_list'] = x['intrinsic_list']
            C = np.round(np.array(x['C']), 2).tolist()
            dp['C'] = C
            dps.append(dp)
            if x_id % 20000 == 0:
                print('tuple collecting', len(scene_cnt.keys()), x_id, cnt_failed)
            # print(x)
        except:
            cnt_failed += 1
        
    split_dps(dps, tgt_name)
    # # json.dump(dps, open("/home/zgtang/scannet_large_easy/dps.json", 'w'), indent=4) # 134000, 1341
    # json.dump(dps, open(f"/home/zgtang/{data_name}/dps.json", 'w'), indent=4)
    # json_strs = [json.dumps(x) for x in dps]
    # ma_len = max(map(len, json_strs))
    # json_strs = np.array(json_strs, dtype='S'+str(ma_len))
    # with h5py.File(f"/home/zgtang/{data_name}/dps.h5", 'w') as f:
    #     f.create_dataset('json_strs', data=json_strs, compression='gzip')
    
    # np.savez_compressed(f"/home/zgtang/{data_name}/dps.npz", json_strs)
    

tuple_n_general_new(args.data_dir, args.tgt_dir)