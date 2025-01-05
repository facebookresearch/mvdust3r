# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .linear_head import LinearPts3d, GSHead
from .dpt_head import create_dpt_head


def head_factory(head_type, output_mode = None, net = None, has_conf=False, skip = False, sh_degree = 0, pts_head_config = {}):
    """" build a prediction head for the decoder 
    """
    if head_type == "GSHead":
        return GSHead(net, skip = skip, sh_degree = sh_degree)
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf, **pts_head_config)
    elif head_type == 'dpt' and output_mode == 'pts3d':
        return create_dpt_head(net, has_conf=has_conf)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
