import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


architectures = dict()

# kpfcn_backbone = [
#     'simple',
#     'resnetb',
#     'resnetb_strided',
#     'resnetb',
#     'resnetb',
#     'resnetb_strided',
#     'resnetb',
#     'resnetb',
#     'resnetb_strided',
#     'resnetb',
#     'resnetb',
#     'nearest_upsample',
#     'unary',
#     'nearest_upsample',
#     'unary',
#     'nearest_upsample',
#     'last_unary'
# ]

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
    'resnetb',
    # 'nearest_upsample',
    # 'unary',
    # 'nearest_upsample',
    # 'unary',
    # 'nearest_upsample',
    # 'last_unary'
]


architectures['KPFCN'] = kpfcn_backbone


KP_arg = add_argument_group('KPConv')
KP_arg.add_argument('--architectures', type=list, default=kpfcn_backbone)
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



def get_kp_config():
  args = parser.parse_args()
  return args