META_INTERNAL=False torchrun --nnodes=1 --nproc_per_node=8 train.py \
--train_dataset " \
15000 @ MVDataset(split='all', ROOT='trajectories/scannet_test', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter, num_views=10, num_render_views=2, random_order = True, random_render_order = False, dps_name = 'dps_test_local.h5', random_nv_nr = [[10,2]], n_ref = 4)" \
--test_dataset " \
MVDataset(n_all=1000, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, num_views=25, num_render_views=5, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_25_5', n_ref = 4) + \
MVDataset(n_all=1000, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, num_views=20, num_render_views=4, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_20_4', n_ref = 4) + \
MVDataset(n_all=1000, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, num_views=30, num_render_views=6, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_30_6', n_ref = 4) + \
MVDataset(n_all=1000, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, num_views=5,  num_render_views=1, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_5_1', n_ref = 4) + \
MVDataset(n_all=1000, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, num_views=10, num_render_views=2, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_10_2', n_ref = 4) + \
MVDataset(n_all=1000, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, num_views=15, num_render_views=3, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_15_3', n_ref = 4) + \
MVDataset(n_all=128, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, save_results=True, save_prefix='test', num_views=30, num_render_views=6, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_25_5_vis', n_ref = 4) + \
MVDataset(n_all=128, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, save_results=True, save_prefix='test', num_views=25, num_render_views=5, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_20_4_vis', n_ref = 4) + \
MVDataset(n_all=128, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, save_results=True, save_prefix='test', num_views=20, num_render_views=4, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_30_6_vis', n_ref = 4) + \
MVDataset(n_all=128, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, save_results=True, save_prefix='test', num_views=15, num_render_views=3, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_5_1_vis', n_ref = 4) + \
MVDataset(n_all=128, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, save_results=True, save_prefix='test', num_views=10, num_render_views=2, render_start = 24, dps_name = 'dps_test_local.h5', tb_name = 'sc_10_2_vis', n_ref = 4) + \
MVDataset(n_all=128, split='all', ROOT='trajectories/scannet_test', resolution=224, seed=777, fix_order=True, save_results=True, save_prefix='test', num_views= 5, num_render_views=1, render_start = 24, dps_name = 'dps_test_local.h5', tb_name =  'sc_15_3_vis', n_ref = 4)" \
--model "AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True}, m_ref_flag=True)" \
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis', mv = True), alpha=0.2)" \
--test_criterion "Regr3D_ScaleShiftAllInv(L21, gt_scale=True, mv = True)" \
--pretrained "checkpoints/MVDp_s1.pth" \
--batch_size 2 --num_workers 16 --only_test 1 \
--output_dir outputs/MVDp_s1_test
