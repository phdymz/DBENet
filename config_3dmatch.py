import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='HardestContrastiveLossTrainer')
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
trainer_arg.add_argument('--batch_size', type=int, default=2)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

# Hard negative mining
trainer_arg.add_argument('--use_hard_negative', type=str2bool, default=True)
trainer_arg.add_argument('--hard_negative_sample_ratio', type=int, default=0.05)
trainer_arg.add_argument('--hard_negative_max_num', type=int, default=3000)
trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=256)

# Metric learning loss
trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--neg_weight', type=float, default=1)

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

trainer_arg.add_argument('--stat_freq', type=int, default=40)
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)

trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)

# Triplets
trainer_arg.add_argument('--triplet_num_pos', type=int, default=256)
trainer_arg.add_argument('--triplet_num_hn', type=int, default=512)
trainer_arg.add_argument('--triplet_num_rand', type=int, default=1024)

# dNetwork specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='ResUNetBN2C')
net_arg.add_argument('--model_n_out', type=int, default=32, help='Feature dimension')
net_arg.add_argument('--conv1_kernel_size', type=int, default=5)
net_arg.add_argument('--normalize_feature', type=str2bool, default=True)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='feat_match_ratio',help='[feat_match_ratio,rre,rte]')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=200)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
icp_path = "/DISK/qwt/datasets/kitti/data_odometry_velodyne/dataset/icp"
opt_arg.add_argument('--icp_cache_path', type=str, default=icp_path)

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--teacher_weight', type=str, default='/home/ymz/桌面/Distillation/3DMatch.pth')


misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument('--nn_max_n',
                      type=int,
                      default=500,
                      help='The maximum number of features to find nearest neighbors in batch')

# Dataset specific configurations
data_arg = add_argument_group('Data')
# ----------------------------------------------------------------------- #
# 3DMatch ---- |output path|
output_3DMatch = "../outputs"
logging_arg.add_argument('--out_dir', type=str, default=output_3DMatch)

# 3DMatch ---- |resume dir|
misc_arg.add_argument('--resume_dir', type=str, default=None)

# 3DMtach ---- |num thread|
misc_arg.add_argument('--train_num_thread', type=int, default=8)
misc_arg.add_argument('--val_num_thread', type=int, default=8)
misc_arg.add_argument('--test_num_thread', type=int, default=8)

# 3DMatch ---- |dataset|
dataset_3DMatch = 'ThreeDMatchPairDataset'
data_arg.add_argument('--dataset', type=str, default=dataset_3DMatch)

# 3DMatch ---- |voxel size|
voxel_size_3DMatch = 0.025
# voxel_size_3DMatch = 0.05
data_arg.add_argument('--voxel_size', type=float, default=voxel_size_3DMatch)
# ----------------------------------------------------------------------- #

# Dataset path
# data_path = "/DISK/qwt/datasets/Ours_train_0_01/train"
data_path = "/media/ymz/2b933929-0294-4162-9385-4fe3eec72189/distillation/3DImageMatch/3DImageMatch/train"

data_arg.add_argument('--threed_match_dir', type=str, default=data_path)
# overlap_path = "/DISK/qwt/datasets/Ours_train_0_01/overlap"
overlap_path = '/media/ymz/2b933929-0294-4162-9385-4fe3eec72189/distillation/3DImageMatch/3DImageMatch/overlap'
data_arg.add_argument('--overlap_path', type=str, default=overlap_path)

# image setting
data_arg.add_argument('--image_W', type=str, default=160)
data_arg.add_argument('--image_H', type=str, default=120)

kitti_path = "/DISK/qwt/datasets/kitti/data_odometry_velodyne"
data_arg.add_argument('--kitti_root', type=str, default=kitti_path)
data_arg.add_argument(
    '--kitti_max_time_diff',
    type=int,
    default=3,
    help='max time difference between pairs (non inclusive)')
data_arg.add_argument('--kitti_date', type=str, default='2020_09_30')


#kpconv
architectures = dict()

kpfcn_backbone = [
    'simple',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'last_unary',
    # 'nearest_upsample',
    # 'unary',
    # 'nearest_upsample',
    # 'unary',
    # 'nearest_upsample',
    # 'last_unary'
]


architectures['KPFCN'] = kpfcn_backbone


KP_arg = add_argument_group('KPConv')
KP_arg.add_argument('--architecture', type=list, default=kpfcn_backbone)
KP_arg.add_argument('--num_layers', type=int, default=4)
KP_arg.add_argument('--in_points_dim', type=int, default=3)
KP_arg.add_argument('--first_feats_dim', type=int, default=128)
KP_arg.add_argument('--final_feats_dim', type=int, default=32)
KP_arg.add_argument('--first_subsampling_dl', type=float, default=0.025)
KP_arg.add_argument('--in_feats_dim', type=int, default=1)
KP_arg.add_argument('--conv_radius', type=float, default=2.5)
KP_arg.add_argument('--deform_radius', type=float, default=5.0)
KP_arg.add_argument('--num_kernel_points', type=int, default=15)
KP_arg.add_argument('--KP_extent', type=float, default=2.0)
KP_arg.add_argument('--batch_norm_momentum', type=float, default=0.02)
KP_arg.add_argument('--use_batch_norm', type=bool, default=True)
KP_arg.add_argument('--add_cross_score', type=bool, default=True)
KP_arg.add_argument('--condition_feature', type=bool, default=True)
KP_arg.add_argument('--deformable', type=bool, default=False)
KP_arg.add_argument('--modulated', type=bool, default=False)
KP_arg.add_argument('--KP_influence', type=str, default='linear')
KP_arg.add_argument('--aggregation_mode', type=str, default='sum')
KP_arg.add_argument('--fixed_kernel_points', type=str, default='center')

Distillation = add_argument_group('Distillation')
Distillation.add_argument('--w_kd', type=float, default=0.0, help='weight for distillation loss')
Distillation.add_argument('--initial_fcgf', type=bool, default=True, help='initalize the weight of fcgf backbone')
Distillation.add_argument('--temperature', type=float, default=1.0)
Distillation.add_argument('--use_softmax', type=bool, default=True, help='use softmax in kd')








def get_config():
  args = parser.parse_args()
  return args
