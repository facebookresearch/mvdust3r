# Copyright (C) 2025-present Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC BY-NC 4.0 (non-commercial use only).


from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed
from dust3r.losses import swap, swap_ref

import dust3r.utils.path_to_croco
from models.croco import CroCoNet

import  torch.nn as nn

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2

def except_i(a, i):

    if i == 0:
        return a[1:]
    elif i == len(a) - 1:
        return a[:-1]
    if type(a) == list:
        return a[:i] + a[i+1:]
    return torch.cat([a[:i], a[i+1:]], dim=0)

class AsymmetricCroCo3DStereoMultiView (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, # # AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed (for non-square images)
                 GS = False,
                 GS_skip = False,
                 sh_degree = 0,
                 pts_head_config = {},
                 n_ref = None,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)
        # dust3r specific initialization
        self.pts_head_config = pts_head_config
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.GS = GS
        self.GS_skip = GS_skip
        self.sh_degree = sh_degree
        self.n_ref = n_ref
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path): # here
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze): # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), pts_head_config = self.pts_head_config)
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), pts_head_config = self.pts_head_config)
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        self.GS_head = [None, None]
        if self.GS:
            self.downstream_GS_head = nn.ModuleList([head_factory("GSHead", net = self, skip = self.GS_skip, sh_degree = self.sh_degree) for i in range(2)])
            self.GS_head = [transpose_to_landscape(self.downstream_GS_head[i], activate=landscape_only) for i in range(2)]
            
    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape) # pos is (x,y) location pair used for rope.

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2s, true_shape1, true_shape2s):
        if img1.shape[-2:] == img2s[0].shape[-2:]:
            n_view = 1 + len(img2s)
            out, pos, _ = self._encode_image(torch.cat((img1, *img2s), dim=0),
                                             torch.cat((true_shape1, *true_shape2s), dim=0))
            outs = out.chunk(n_view, dim=0)
            poss = pos.chunk(n_view, dim=0)
            out, out2s = outs[0], outs[1:]
            pos, pos2s = poss[0], poss[1:]
        else:
            raise NotImplementedError
        return out, out2s, pos, pos2s

    def _encode_symmetrized(self, view1, view2s):
        img1 = view1['img']
        img2s = [view2['img'] for view2 in view2s]
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2s = [view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1)) for (view2, img2) in zip(view2s, img2s)]
        # warning! maybe the images have different portrait/landscape orientations

        feat1, feat2s, pos1, pos2s = self._encode_image_pairs(img1, img2s, shape1, shape2s)

        return (shape1, shape2s), (feat1, feat2s), (pos1, pos2s)

    def _decoder(self, f1, pos1, f2s, pos2s, n_ref = 1):
        if n_ref > 1:
            return self._decoder_multi_ref(f1, pos1, f2s, pos2s, n_ref)
        final_output = [(f1, *f2s)]  # before projection
        n_view_src = len(f2s)
        # project to decoder dim
        f1 = self.decoder_embed(f1) # [bs, 14 * 14, 1024] -> [bs, 14 * 14, 768]
        bs = f1.shape[0]
        f2s = torch.cat(f2s, dim = 0)
        f2s = self.decoder_embed(f2s).split(bs)

        final_output.append((f1, *f2s))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(final_output[-1][0], final_output[-1][1:], pos1, pos2s, mv = True)
            # img2 side
            f2s = []
            for i in range(n_view_src):
                f2s_other = list(final_output[-1][:1 + i]) + list(final_output[-1][1 + i + 1:])
                pos2s_other = [pos1] + list(pos2s[:i]) + list(pos2s[i+1:])
                f2s.append(blk2(final_output[-1][1 + i], f2s_other, pos2s[i], pos2s_other, mv = True)[0]) # TODO: here maybe we need distinguish the ref 
            # store the result
            final_output.append((f1, *f2s))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))

        f1_all = []
        f2_alls = [[] for i in range(n_view_src)]
        for i in range(len(final_output)):
            f1_all.append(final_output[i][0])
            for j in range(n_view_src):
                f2_alls[j].append(final_output[i][1 + j])
        return f1_all, f2_alls

    def _decoder_multi_ref(self, f1, pos1, f2s, pos2s, n_ref = 1):
        final_output_mref = [[[f1, *f2s] for i in range(n_ref)]]
        n_view_src = len(f2s)
        nv = n_view_src + 1
        # project to decoder dim
        f1 = self.decoder_embed(f1) # [bs, 14 * 14, 1024] -> [bs, 14 * 14, 768]
        bs = f1.shape[0]
        f2s = torch.cat(f2s, dim = 0)
        f2s = self.decoder_embed(f2s).split(bs) # nv of [bs, 14 * 14, 768]
        pos_all = [pos1] + list(pos2s)
        final_output_mref.append([[f1, *f2s] for i in range(n_ref)])
        for blk1, blk2, blk_sv in zip(self.dec_blocks, self.dec_blocks2, self.dec_same_view_blocks):
            final_output_mref_i = []
            for ref_id in range(n_ref):
                # img1 side
                fs = [None for i in range(nv)]
                f1, _ = blk1(final_output_mref[-1][ref_id][ref_id], except_i(final_output_mref[-1][ref_id], ref_id), pos1, pos2s, mv = True) # def forward(self, x, y, xpos, ypos, mv = False):
                fs[ref_id] = f1
                # img2 side
                for other_view_id in range(nv):
                    if other_view_id == ref_id:
                        continue
                    # f2s_other = list(final_output[-1][:1 + i]) + list(final_output[-1][1 + i + 1:])
                    # pos2s_other = [pos1] + list(pos2s[:i]) + list(pos2s[i+1:])
                    f2 = blk2(final_output_mref[-1][ref_id][other_view_id], except_i(final_output_mref[-1][ref_id], other_view_id), pos1, pos2s, mv = True)[0] # TODO: here maybe we need distinguish the ref (pos should not be simply "pos1, pose2s"), but pos are the same for all views in the current implementation, need change later. 
                    fs[other_view_id] = f2
                # store the result
                final_output_mref_i.append(fs)
            fs_new = [[None for i in range(nv)] for j in range(n_ref)] # [n_ref, nv, bs, 14 * 14, 768]
            for view_id in range(nv):
                final_output_mref_i_view = [final_output_mref_i[i][view_id] for i in range(n_ref)]
                for ref_id in range(n_ref):
                    if blk_sv is not None:
                        fs_new[ref_id][view_id] = blk_sv(final_output_mref_i[ref_id][view_id], except_i(final_output_mref_i_view, ref_id), pos1, pos2s[:n_ref - 1], mv = True, coeff = 1.)[0] # debug
                    else:
                        fs_new[ref_id][view_id] = final_output_mref_i[ref_id][view_id]
                    
            final_output_mref.append(fs_new)

        # normalize last output
        del final_output_mref[1]  # duplicate with final_output[0]
        final_output = final_output_mref
        # bs * n_ref
        final_output_last = []
        for view_id in range(nv):
            final_output_last_view = torch.stack([final_output[-1][i][view_id] for i in range(n_ref)], dim = 1) # [bs, n_ref, 14 * 14, 768]
            final_output_last.append(final_output_last_view.flatten(0, 1)) # nv of [bs * n_ref, 14 * 14, 768]
        final_output[-1] = tuple(final_output_last)

        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))

        for data_id in range(bs):
            for ref_id in range(1, n_ref):
                swap_ref(final_output[-1][0][data_id * n_ref + ref_id], final_output[-1][ref_id][data_id * n_ref + ref_id])
        
        final_output = final_output[-1:]
        f1_all = []
        f2_alls = [[] for i in range(n_view_src)]
        for i in range(len(final_output)):
            f1_all.append(final_output[i][0])
            for j in range(n_view_src):
                f2_alls[j].append(final_output[i][1 + j])
        
        return f1_all, f2_alls

    def _downstream_head(self, head_num, decout, img_shape):
        # B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)
    
    def _downstream_head_GS(self, head_num, decout, img_shape):
        # B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = self.GS_head[head_num - 1]
        return head(decout, img_shape)

    def forward(self, view1, view2s_all):
        # encode the two images --> B,S,D
        num_render_views = view2s_all[0].get("num_render_views", torch.Tensor([0]).long())[0].item()
        n_ref = view2s_all[0].get("n_ref", torch.Tensor([1]).long())[0].item()
        if self.n_ref is not None:
            n_ref = self.n_ref
        assert self.m_ref_flag == False or (self.m_ref_flag == True and n_ref > 1), f"No. of reference views should be > 1 if m_ref_flag is True"

        if num_render_views:
            view2s, view2s_render = view2s_all[:-num_render_views], view2s_all[-num_render_views:]
        else:
            view2s, view2s_render = view2s_all, []
        
        (shape1, shape2s), (feat1, feat2s), (pos1, pos2s) = self._encode_symmetrized(view1, view2s) # every view is dealt with the same param.
        
        # combine all ref images into object-centric representation
        dec1, dec2s = self._decoder(feat1, pos1, feat2s, pos2s, n_ref = n_ref)
        
        with torch.cuda.amp.autocast(enabled=False):
            # print('1 shape', [tok.shape for tok in dec1]) # 1 shape [torch.Size([4, 14 * 14, 1024]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768])]
            # print('2 shape', [[tok.shape for tok in dec2] for (dec2, shape2) in zip(dec2s, shape2s)]) # 2 shape [[torch.Size([4, 196, 1024]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768])], [torch.Size([4, 196, 1024]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768])], [torch.Size([4, 196, 1024]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768]), torch.Size([4, 196, 768])]]
            bs = view1['img'].shape[0]
            view1_img = view1['img'].repeat_interleave(n_ref, dim = 0)
            view2s_img = [view2['img'].repeat_interleave(n_ref, dim = 0) for view2 in view2s]
            
            views_img = [view1_img] + view2s_img
            for data_id in range(bs):
                for ref_id in range(1, n_ref):
                    swap_ref(views_img[0][data_id * n_ref + ref_id], views_img[ref_id][data_id * n_ref + ref_id])
            view1_img = views_img[0]
            view2s_img = views_img[1:]

            res1 = self._downstream_head(1, ([tok.float() for tok in dec1], view1_img), shape1)
            res2s = [self._downstream_head(2, ([tok.float() for tok in dec2], view2_img), shape2) for (dec2, shape2, view2_img) in zip(dec2s, shape2s, view2s_img)]
            if self.GS:
                res1_GS = self._downstream_head_GS(1, ([tok.float() for tok in dec1], view1_img), shape1)
                res2s_GS = [self._downstream_head_GS(2, ([tok.float() for tok in dec2], view2_img), shape2) for (dec2, shape2, view2_img) in zip(dec2s, shape2s, view2s_img)]
                res1 = {**res1, **res1_GS}
                res2s_new = []
                for (res2, res2_GS) in zip(res2s, res2s_GS):
                    res2 = {**res2, **res2_GS}
                    res2s_new.append(res2)
                res2s = res2s_new

            for res2 in res2s:
                res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        
        return res1, res2s