# Copyright (C) 2025-present Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC BY-NC 4.0 (non-commercial use only).


import os

def get_local_path(x):
    return x

def set_device(x):
    return

class g_pathmgr:
    open=open
    get_local_path=get_local_path

    @classmethod
    def mkdirs(cls, value):
        return os.makedirs(value, exist_ok=True)
    
    @classmethod
    def isfile(cls, value):
        return os.path.isfile(value)

def get_log_dir_warp(output_dir):

    return output_dir


def change_to_sr(file_lists):
    return file_lists

def save_image_manifold(img, tgt):

    import imageio
    imageio.imwrite(tgt, img)
