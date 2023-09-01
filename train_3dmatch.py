import open3d as o3d  # prevent loading error
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import logging
import torch
from easydict import EasyDict as edict
import warnings
warnings.filterwarnings("ignore")

from lib.data_loaders import make_data_loader, get_dataset
from config_3dmatch import get_config as get_config_3dmatch
from config.kpfcn_config import get_kp_config
from lib.kpsampling import get_dataloader

from lib.trainer import Distillation, Distillation_KP, Distillation_frozen_encoder


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


# def get_trainer(trainer):
#   if trainer == 'ContrastiveLossTrainer':
#     return ContrastiveLossTrainer
#   elif trainer == 'HardestContrastiveLossTrainer':
#     return HardestContrastiveLossTrainer
#   else:
#     raise ValueError(f'Trainer {trainer} not found')


def main(config, resume=False):
    train_set = get_dataset(
        config,
        config.train_phase,
        )
    train_loader, neighborhood_limits = get_dataloader(
      train_set,
      batch_size=config.batch_size,
      num_workers=config.train_num_thread,
      shuffle=True,
      neighborhood_limits=None,
      )

    val_set = get_dataset(
        config,
        config.val_phase,
        )
    val_loader, _ = get_dataloader(
      val_set,
      batch_size=config.val_batch_size,
      num_workers=config.val_num_thread,
      shuffle=True,
      neighborhood_limits=neighborhood_limits,
      )

    # Trainer = get_trainer(config.trainer)
    trainer = Distillation_KP(
          config=config,
          data_loader=train_loader,
          val_data_loader=val_loader,
      )

    trainer.train()


if __name__ == "__main__":
  logger = logging.getLogger()
  config = get_config_3dmatch()

  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  main(config)