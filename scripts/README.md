# Important hyperparameter explanations

## MVDataset

n_all: No. of data samples uniformly chosen from the whole h5 list.
num_views: No. of input views + novel views 
num_render_views: No. of novel views (then we will have num_input_views = num_views - num_render_views)
random_order: whether data augmentation of random shuffling all images happens.
random_nv_nr: a list of pairs (num_views, num_render_views). this will lead to mixed length training. (for MVDp+'s stage 2)
n_ref: number of reference views (and MV-DUSt3R+ will take them for multiple paths)

## Model

pts_head_config: skip: whether there is a skip layer in the end of linear layer for higher frequency short path.
m_ref_flag: whether it's MV-DUSt3R or MV-DUSt3R+



