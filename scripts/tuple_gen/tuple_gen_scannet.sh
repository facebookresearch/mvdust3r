n_job=4
n_v=15
n_render=3
data_type=scannet
hardness=easy
split=test
n_tuple_per_scene=1000

for ((i=0; i<n_job; i++))
do
  echo $((8 * n_job)) $i
  META_INTERNAL=False torchrun --nnodes=1 --nproc_per_node=8 datasets_preprocess/scannet_tuple_gen.py -- --div $((8 * n_job)) --node-no $i --n-v $n_v --n-render $n_render --data-type $data_type --hardness $hardness --split $split --n-tuple-per-scene $n_tuple_per_scene &
done



n_job=4
n_v=15
n_render=3
data_type=scannet
hardness=easier
split=all
n_tuple_per_scene=1000

for ((i=0; i<n_job; i++))
do
  echo $((8 * n_job)) $i
  META_INTERNAL=False torchrun --nnodes=1 --nproc_per_node=8 datasets_preprocess/scannet_tuple_gen.py -- --div $((8 * n_job)) --node-no $i --n-v $n_v --n-render $n_render --data-type $data_type --hardness $hardness --split $split --n-tuple-per-scene $n_tuple_per_scene &
done



