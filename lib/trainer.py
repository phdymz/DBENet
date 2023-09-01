import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash

import MinkowskiEngine as ME
from model.kpfcgf import ResUNetBN3C, ResUNetBN4C, ResUNetBN5C
from lib.loss import CriterionKD



class AlignmentTrainer:

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3,
        config=config)

    if config.weights:
      checkpoint = torch.load(config.weights)
      model.load_state_dict(checkpoint['state_dict'])

    logging.info(model)

    self.config = config
    self.model = model
    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf
    if(self.best_val_metric == "rre" or self.best_val_metric == "rte"):
      self.best_val = np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.optimizer = getattr(optim, config.optimizer)(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    # self.test_valid = True if self.val_data_loader is not None else False
    self.test_valid = config.test_valid
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        self.start_epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")
  # ---- train start ----
  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()
          self._save_checkpoint(
            epoch=epoch,
            val_dict=val_dict,
            best_val_metric=self.best_val_metric
          )

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if(self.best_val_metric == "feat_match_ratio" or self.best_val_metric == "success"):
          if self.best_val < val_dict[self.best_val_metric]:
            logging.info(
                f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
                f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )
        elif(self.best_val_metric == "rre" or self.best_val_metric == "rte" ):
          if self.best_val > val_dict[self.best_val_metric]:
            logging.info(
                f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
                f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )

  def _save_checkpoint(self, epoch ,val_dict,best_val_metric,filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    content = f"{best_val_metric}_{val_dict[best_val_metric]}"
    filename = os.path.join(self.checkpoint_dir, f'{filename}_epoch_{epoch}_{content}.pth')

    logging.info(f"Saving checkpoint: {filename} ...")
    torch.save(state, filename)


class ContrastiveLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    # if val_data_loader is not None:
    #   assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        # Caffe iter size
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # pairs consist of (xyz1 index, xyz0 index)
        image0 = input_dict['image0'].to(self.device)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'].to(self.device),
            coordinates=input_dict['sinput0_C'].to(self.device))
        F0 = self.model(sinput0,image0).F

        image1 = input_dict['image1'].to(self.device)
        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'].to(self.device),
            coordinates=input_dict['sinput1_C'].to(self.device))
        F1 = self.model(sinput1,image1).F

        N0, N1 = len(sinput0), len(sinput1)

        pos_pairs = input_dict['correspondences']
        neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)
        pos_pairs = pos_pairs.long().to(self.device)
        neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)

        neg0 = F0.index_select(0, neg_pairs[:, 0])
        neg1 = F1.index_select(0, neg_pairs[:, 1])
        pos0 = F0.index_select(0, pos_pairs[:, 0])
        pos1 = F1.index_select(0, pos_pairs[:, 1])

        # Positive loss
        pos_loss = (pos0 - pos1).pow(2).sum(1)

        # Negative loss
        neg_loss = F.relu(self.neg_thresh -
                          ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size

        # Weighted loss
        loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the begining of iter_size
        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

      self.optimizer.step()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  # ---- current valid ----
  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter,success_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(),AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      image0 = input_dict['image0'].to(self.device)
      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'].to(self.device),
          coordinates=input_dict['sinput0_C'].to(self.device))
      F0 = self.model(sinput0,image0).F

      image1 = input_dict['image1'].to(self.device)
      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'].to(self.device),
          coordinates=input_dict['sinput1_C'].to(self.device))
      F1 = self.model(sinput1,image1).F
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
      if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
        success_meter.update(1)
      else:
        success_meter.update(0)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f}, Success:{success_meter.avg:.3f}",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
        ]))
        data_timer.reset()

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg,
        "success":success_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()


class HardestContrastiveLossTrainer(ContrastiveLossTrainer):

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  # ---- current train ----
  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        image0 = input_dict['image0'].to(self.device)
        sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'].to(self.device),
            coordinates=input_dict['sinput0_C'].to(self.device))
        F0 = self.model(sinput0,image0).F

        image1 = input_dict['image1'].to(self.device)
        sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'].to(self.device),
            coordinates=input_dict['sinput1_C'].to(self.device))
        F1 = self.model(sinput1,image1).F

        pos_pairs = input_dict['correspondences']
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=self.config.num_pos_per_batch * self.config.batch_size,
            num_hn_samples=self.config.num_hn_samples_per_batch *
            self.config.batch_size)

        pos_loss /= iter_size
        neg_loss /= iter_size
        loss = pos_loss + self.neg_weight * neg_loss
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()


class Scratch:
    def __init__(self, config, data_loader, val_data_loader):

      num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

      # Model initialization
      Model = ResUNetBN3C
      model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3,
        config=config)

      if config.weights:
        checkpoint = torch.load(config.weights)
        model.load_state_dict(checkpoint['state_dict'])

      logging.info(model)

      self.config = config
      self.model = model
      self.max_epoch = config.max_epoch
      self.save_freq = config.save_freq_epoch
      self.val_max_iter = config.val_max_iter
      self.val_epoch_freq = config.val_epoch_freq

      self.best_val_metric = config.best_val_metric
      self.best_val_epoch = -np.inf
      self.best_val = -np.inf
      if (self.best_val_metric == "rre" or self.best_val_metric == "rte"):
        self.best_val = np.inf

      if config.use_gpu and not torch.cuda.is_available():
        logging.warning('Warning: There\'s no CUDA support on this machine, '
                        'training is performed on CPU.')
        raise ValueError('GPU not available, but cuda flag set')

      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      self.optimizer = getattr(optim, config.optimizer)(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

      self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

      self.start_epoch = 1
      self.checkpoint_dir = config.out_dir

      ensure_dir(self.checkpoint_dir)
      json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

      self.iter_size = config.iter_size
      self.batch_size = data_loader.batch_size
      self.data_loader = data_loader
      self.val_data_loader = val_data_loader

      # self.test_valid = True if self.val_data_loader is not None else False
      self.test_valid = config.test_valid
      self.log_step = int(np.sqrt(self.config.batch_size))
      self.model = self.model.to(self.device)
      self.writer = SummaryWriter(logdir=config.out_dir)

      if config.resume is not None:
        if osp.isfile(config.resume):
          logging.info("=> loading checkpoint '{}'".format(config.resume))
          state = torch.load(config.resume)
          self.start_epoch = state['epoch']
          model.load_state_dict(state['state_dict'])
          self.scheduler.load_state_dict(state['scheduler'])
          self.optimizer.load_state_dict(state['optimizer'])

          if 'best_val' in state.keys():
            self.best_val = state['best_val']
            self.best_val_epoch = state['best_val_epoch']
            self.best_val_metric = state['best_val_metric']
        else:
          raise ValueError(f"=> no checkpoint found at '{config.resume}'")

      self.neg_thresh = config.neg_thresh
      self.pos_thresh = config.pos_thresh
      self.neg_weight = config.neg_weight

      # self.kp_list = ['points', 'neighbors', 'pools', 'upsamples', 'features', 'stack_lengths']
      # ---- train start ----

    def train(self):
      """
      Full training logic
      """
      # Baseline random feature performance
      if self.test_valid:
        with torch.no_grad():
          val_dict = self._valid_epoch()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, 0)

      for epoch in range(self.start_epoch, self.max_epoch + 1):
        lr = self.scheduler.get_lr()
        logging.info(f" Epoch: {epoch}, LR: {lr}")
        self._train_epoch(epoch)
        self.scheduler.step()

        if self.test_valid and epoch % self.val_epoch_freq == 0:
          with torch.no_grad():
            val_dict = self._valid_epoch()
            self._save_checkpoint(
              epoch=epoch,
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )

          for k, v in val_dict.items():
            self.writer.add_scalar(f'val/{k}', v, epoch)
          if (self.best_val_metric == "feat_match_ratio" or self.best_val_metric == "success"):
            if self.best_val < val_dict[self.best_val_metric]:
              logging.info(
                f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
              )
              self.best_val = val_dict[self.best_val_metric]
              self.best_val_epoch = epoch
              self._save_checkpoint(
                epoch=epoch,
                filename='best_val_checkpoint',
                val_dict=val_dict,
                best_val_metric=self.best_val_metric
              )
            else:
              logging.info(
                f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
              )
          elif (self.best_val_metric == "rre" or self.best_val_metric == "rte"):
            if self.best_val > val_dict[self.best_val_metric]:
              logging.info(
                f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
              )
              self.best_val = val_dict[self.best_val_metric]
              self.best_val_epoch = epoch
              self._save_checkpoint(
                epoch=epoch,
                filename='best_val_checkpoint',
                val_dict=val_dict,
                best_val_metric=self.best_val_metric
              )
            else:
              logging.info(
                f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
              )

    def _save_checkpoint(self, epoch, val_dict, best_val_metric, filename='checkpoint'):
      state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
      }
      content = f"{best_val_metric}_{val_dict[best_val_metric]}"
      filename = os.path.join(self.checkpoint_dir, f'{filename}_epoch_{epoch}_{content}.pth')

      logging.info(f"Saving checkpoint: {filename} ...")
      torch.save(state, filename)

    def apply_transform(self, pts, trans):
      R = trans[:3, :3]
      T = trans[:3, 3]
      return pts @ R.t() + T

    def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
      """
      Generate random negative pairs
      """
      if not isinstance(positive_pairs, np.ndarray):
        positive_pairs = np.array(positive_pairs, dtype=np.int64)
      if N_neg < 1:
        N_neg = positive_pairs.shape[0] * 2
      pos_keys = _hash(positive_pairs, hash_seed)

      neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
      neg_keys = _hash(neg_pairs, hash_seed)
      mask = np.isin(neg_keys, pos_keys, assume_unique=False)
      return neg_pairs[np.logical_not(mask)]

    def contrastive_hardest_negative_loss(self,
                                          F0,
                                          F1,
                                          positive_pairs,
                                          num_pos=5192,
                                          num_hn_samples=2048,
                                          thresh=None):
      """
      Generate negative pairs
      """
      N0, N1 = len(F0), len(F1)
      N_pos_pairs = len(positive_pairs)
      hash_seed = max(N0, N1)
      sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
      sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

      if N_pos_pairs > num_pos:
        pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
        sample_pos_pairs = positive_pairs[pos_sel]
      else:
        sample_pos_pairs = positive_pairs

      # Find negatives for all F1[positive_pairs[:, 1]]
      subF0, subF1 = F0[sel0], F1[sel1]

      pos_ind0 = sample_pos_pairs[:, 0].long()
      pos_ind1 = sample_pos_pairs[:, 1].long()
      posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

      D01 = pdist(posF0, subF1, dist_type='L2')
      D10 = pdist(posF1, subF0, dist_type='L2')

      D01min, D01ind = D01.min(1)
      D10min, D10ind = D10.min(1)

      if not isinstance(positive_pairs, np.ndarray):
        positive_pairs = np.array(positive_pairs, dtype=np.int64)

      pos_keys = _hash(positive_pairs, hash_seed)

      D01ind = sel1[D01ind.cpu().numpy()]
      D10ind = sel0[D10ind.cpu().numpy()]
      neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
      neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

      mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
      mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
      pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
      neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
      neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
      return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    # ---- current train ----
    def _train_epoch(self, epoch):
      gc.collect()
      self.model.train()
      # Epoch starts from 1
      total_loss = 0
      total_num = 0.0
      data_loader = self.data_loader
      data_loader_iter = self.data_loader.__iter__()
      iter_size = self.iter_size
      data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
      start_iter = (epoch - 1) * (len(data_loader) // iter_size)
      for curr_iter in range(len(data_loader) // iter_size):
        self.optimizer.zero_grad()
        batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

        data_time = 0
        total_timer.tic()
        for iter_idx in range(iter_size):
          data_timer.tic()
          input_dict = data_loader_iter.next()
          data_time += data_timer.toc(average=False)

          # image0 = input_dict['image0'].to(self.device)
          # image1 = input_dict['image1'].to(self.device)

          batch0 = input_dict['batch0']
          for k, v in batch0.items():
            if type(v) == list:
              batch0[k] = [item.to(self.device) for item in v]
            elif type(v) == dict:
              pass
            else:
              batch0[k] = v.to(self.device)
          batch1 = input_dict['batch1']
          for k, v in batch1.items():
            if type(v) == list:
              batch1[k] = [item.to(self.device) for item in v]
            elif type(v) == dict:
              pass
            else:
              batch1[k] = v.to(self.device)


          sinput0 = ME.SparseTensor(
            input_dict['sinput0_F'].to(self.device),
            coordinates=input_dict['sinput0_C'].to(self.device))
          F0 = self.model(sinput0, batch0).F


          sinput1 = ME.SparseTensor(
            input_dict['sinput1_F'].to(self.device),
            coordinates=input_dict['sinput1_C'].to(self.device))
          F1 = self.model(sinput1, batch1).F

          pos_pairs = input_dict['correspondences']
          pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=self.config.num_pos_per_batch * self.config.batch_size,
            num_hn_samples=self.config.num_hn_samples_per_batch *
                           self.config.batch_size)

          pos_loss /= iter_size
          neg_loss /= iter_size
          loss = pos_loss + self.neg_weight * neg_loss
          loss.backward()

          batch_loss += loss.item()
          batch_pos_loss += pos_loss.item()
          batch_neg_loss += neg_loss.item()

        self.optimizer.step()
        gc.collect()

        torch.cuda.empty_cache()

        total_loss += batch_loss
        total_num += 1.0
        total_timer.toc()
        data_meter.update(data_time)

        if curr_iter % self.config.stat_freq == 0:
          self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
          self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
          self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
          logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
              data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
          data_meter.reset()
          total_timer.reset()

    # ---- current valid ----
    def _valid_epoch(self):
      # Change the network to evaluation mode
      self.model.eval()
      self.val_data_loader.dataset.reset_seed(0)
      num_data = 0
      hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, success_meter = AverageMeter(
      ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
      data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
      tot_num_data = len(self.val_data_loader.dataset)
      if self.val_max_iter > 0:
        tot_num_data = min(self.val_max_iter, tot_num_data)
      data_loader_iter = self.val_data_loader.__iter__()

      for batch_idx in range(tot_num_data):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_timer.toc()

        # pairs consist of (xyz1 index, xyz0 index)
        feat_timer.tic()
        # image0 = input_dict['image0'].to(self.device)
        # sinput0 = ME.SparseTensor(
        #   input_dict['sinput0_F'].to(self.device),
        #   coordinates=input_dict['sinput0_C'].to(self.device))
        # F0 = self.model(sinput0, image0).F
        #
        # image1 = input_dict['image1'].to(self.device)
        # sinput1 = ME.SparseTensor(
        #   input_dict['sinput1_F'].to(self.device),
        #   coordinates=input_dict['sinput1_C'].to(self.device))
        # F1 = self.model(sinput1, image1).F
        batch0 = input_dict['batch0']
        for k, v in batch0.items():
          if type(v) == list:
            batch0[k] = [item.to(self.device) for item in v]
          elif type(v) == dict:
            pass
          else:
            batch0[k] = v.to(self.device)
        batch1 = input_dict['batch1']
        for k, v in batch1.items():
          if type(v) == list:
            batch1[k] = [item.to(self.device) for item in v]
          elif type(v) == dict:
            pass
          else:
            batch1[k] = v.to(self.device)

        sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'].to(self.device),
          coordinates=input_dict['sinput0_C'].to(self.device))
        F0 = self.model(sinput0, batch0).F

        sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'].to(self.device),
          coordinates=input_dict['sinput1_C'].to(self.device))
        F1 = self.model(sinput1, batch1).F
        feat_timer.toc()

        matching_timer.tic()
        xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
        xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
        T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

        loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
        loss_meter.update(loss)

        rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        rte_meter.update(rte)
        rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
        if not np.isnan(rre):
          rre_meter.update(rre)

        # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
        if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
          success_meter.update(1)
        else:
          success_meter.update(0)

        hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
        hit_ratio_meter.update(hit_ratio)
        feat_match_ratio.update(hit_ratio > 0.05)
        matching_timer.toc()

        num_data += 1
        torch.cuda.empty_cache()

        if batch_idx % 100 == 0 and batch_idx > 0:
          logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f}, Success:{success_meter.avg:.3f}",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
          ]))
          data_timer.reset()

      logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
      ]))
      return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg,
        "success": success_meter.avg
      }

    def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
      subsample = len(F0) > subsample_size
      if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

      # Compute the nn
      nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
      if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
      else:
        return xyz0, xyz1[nn_inds]

    def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
      xyz0 = self.apply_transform(xyz0, T_gth)
      dist = np.sqrt(((xyz0 - xyz1) ** 2).sum(1) + 1e-6)
      return (dist < thresh).float().mean().item()



class Distillation:
  def __init__(self, config, data_loader, val_data_loader):

    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    # student use a kpfcn to learn semantic feature from teacher's resnet34
    student = ResUNetBN4C(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      normalize_feature=config.normalize_feature,
      conv1_kernel_size=config.conv1_kernel_size,
      D=3,
      config=config)

    teacher = ResUNetBN5C(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      normalize_feature=config.normalize_feature,
      conv1_kernel_size=config.conv1_kernel_size,
      D=3,
      config=config)


    checkpoint = torch.load(config.teacher_weight)
    teacher.load_state_dict(checkpoint['state_dict'])

    if config.initial_fcgf:
      miss = student.load_state_dict(checkpoint['state_dict'], strict = False)
      print('miss part:', miss)

    self.config = config
    self.teacher = teacher
    self.student = student

    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf
    if (self.best_val_metric == "rre" or self.best_val_metric == "rte"):
      self.best_val = np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.optimizer = getattr(optim, config.optimizer)(
      self.student.parameters(),
      lr=config.lr,
      momentum=config.momentum,
      weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
      config,
      open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
      indent=4,
      sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    # self.test_valid = True if self.val_data_loader is not None else False
    self.test_valid = config.test_valid
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.teacher = self.teacher.to(self.device)
    self.student = self.student.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    # if config.resume is not None:
    #   if osp.isfile(config.resume):
    #     logging.info("=> loading checkpoint '{}'".format(config.resume))
    #     state = torch.load(config.resume)
    #     self.start_epoch = state['epoch']
    #     model.load_state_dict(state['state_dict'])
    #     self.scheduler.load_state_dict(state['scheduler'])
    #     self.optimizer.load_state_dict(state['optimizer'])
    #
    #     if 'best_val' in state.keys():
    #       self.best_val = state['best_val']
    #       self.best_val_epoch = state['best_val_epoch']
    #       self.best_val_metric = state['best_val_metric']
    #   else:
    #     raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight
    self.w_kd = config.w_kd
    self.kd_loss = CriterionKD(temperature = config.temperature, use_softmax = config.use_softmax)

    # self.kp_list = ['points', 'neighbors', 'pools', 'upsamples', 'features', 'stack_lengths']
    # ---- train start ----

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()
          self._save_checkpoint(
            epoch=epoch,
            val_dict=val_dict,
            best_val_metric=self.best_val_metric
          )

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if (self.best_val_metric == "feat_match_ratio" or self.best_val_metric == "success"):
          if self.best_val < val_dict[self.best_val_metric]:
            logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )
        elif (self.best_val_metric == "rre" or self.best_val_metric == "rte"):
          if self.best_val > val_dict[self.best_val_metric]:
            logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )

  def _save_checkpoint(self, epoch, val_dict, best_val_metric, filename='checkpoint'):
    state = {
      'epoch': epoch,
      'state_dict': self.student.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'scheduler': self.scheduler.state_dict(),
      'config': self.config,
      'best_val': self.best_val,
      'best_val_epoch': self.best_val_epoch,
      'best_val_metric': self.best_val_metric
    }
    content = f"{best_val_metric}_{val_dict[best_val_metric]}"
    filename = os.path.join(self.checkpoint_dir, f'{filename}_epoch_{epoch}_{content}.pth')

    logging.info(f"Saving checkpoint: {filename} ...")
    torch.save(state, filename)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
      np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
      np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
      np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  # ---- current train ----
  def _train_epoch(self, epoch):
    gc.collect()
    self.student.train()
    self.teacher.eval()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss_kd, batch_loss = 0, 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        image0 = input_dict['image0'].to(self.device)
        image1 = input_dict['image1'].to(self.device)

        batch0 = input_dict['batch0']
        for k, v in batch0.items():
          if type(v) == list:
            batch0[k] = [item.to(self.device) for item in v]
          elif type(v) == dict:
            pass
          else:
            batch0[k] = v.to(self.device)
        batch1 = input_dict['batch1']
        for k, v in batch1.items():
          if type(v) == list:
            batch1[k] = [item.to(self.device) for item in v]
          elif type(v) == dict:
            pass
          else:
            batch1[k] = v.to(self.device)

        sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'].to(self.device),
          coordinates=input_dict['sinput0_C'].to(self.device))
        F0, F0_median_s = self.student(sinput0, batch0)
        _, F0_median_t = self.teacher(sinput0, image0)

        sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'].to(self.device),
          coordinates=input_dict['sinput1_C'].to(self.device))
        F1, F1_median_s = self.student(sinput1, batch1)
        _, F1_median_t = self.teacher(sinput1, image1)

        pos_pairs = input_dict['correspondences']
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
          F0,
          F1,
          pos_pairs,
          num_pos=self.config.num_pos_per_batch * self.config.batch_size,
          num_hn_samples=self.config.num_hn_samples_per_batch *
                         self.config.batch_size)

        pos_loss /= iter_size
        neg_loss /= iter_size

        loss_kd = self.distillation_loss(F0_median_s, F0_median_t, F1_median_s, F1_median_t)
        loss_kd /= iter_size

        loss = pos_loss + self.neg_weight * neg_loss + self.w_kd * loss_kd
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()
        batch_loss_kd += loss_kd.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/distillation_loss', batch_loss_kd, start_iter + curr_iter)
        logging.info(
          "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
          .format(epoch, curr_iter,
                  len(self.data_loader) //
                  iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
          "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
            data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  # ---- current valid ----
  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.student.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, success_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      # image0 = input_dict['image0'].to(self.device)
      # sinput0 = ME.SparseTensor(
      #   input_dict['sinput0_F'].to(self.device),
      #   coordinates=input_dict['sinput0_C'].to(self.device))
      # F0 = self.model(sinput0, image0).F
      #
      # image1 = input_dict['image1'].to(self.device)
      # sinput1 = ME.SparseTensor(
      #   input_dict['sinput1_F'].to(self.device),
      #   coordinates=input_dict['sinput1_C'].to(self.device))
      # F1 = self.model(sinput1, image1).F
      batch0 = input_dict['batch0']
      for k, v in batch0.items():
        if type(v) == list:
          batch0[k] = [item.to(self.device) for item in v]
        elif type(v) == dict:
          pass
        else:
          batch0[k] = v.to(self.device)
      batch1 = input_dict['batch1']
      for k, v in batch1.items():
        if type(v) == list:
          batch1[k] = [item.to(self.device) for item in v]
        elif type(v) == dict:
          pass
        else:
          batch1[k] = v.to(self.device)

      sinput0 = ME.SparseTensor(
        input_dict['sinput0_F'].to(self.device),
        coordinates=input_dict['sinput0_C'].to(self.device))
      F0, _ = self.student(sinput0, batch0)

      sinput1 = ME.SparseTensor(
        input_dict['sinput1_F'].to(self.device),
        coordinates=input_dict['sinput1_C'].to(self.device))
      F1, _ = self.student(sinput1, batch1)
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
      if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
        success_meter.update(1)
      else:
        success_meter.update(0)

      hit_ratio = self.evaluate_hit_ratio(
        xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
          f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
          f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
          f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f}, Success:{success_meter.avg:.3f}",
          f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
        ]))
        data_timer.reset()

    logging.info(' '.join([
      f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
      f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
      "loss": loss_meter.avg,
      "rre": rre_meter.avg,
      "rte": rte_meter.avg,
      'feat_match_ratio': feat_match_ratio.avg,
      'hit_ratio': hit_ratio_meter.avg,
      "success": success_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1) ** 2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()

  def distillation_loss(self, F0_median_s, F0_median_t, F1_median_s, F1_median_t):
    loss0 = self.kd_loss(F0_median_s, F0_median_t)
    loss1 = self.kd_loss(F1_median_s, F1_median_t)
    return loss0 + loss1




#only distillation parameter of KPFCN, frozen the backbone of FCGF
class Distillation_KP:
  def __init__(self, config, data_loader, val_data_loader):

    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    # student use a kpfcn to learn semantic feature from teacher's resnet34
    student = ResUNetBN4C(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      normalize_feature=config.normalize_feature,
      conv1_kernel_size=config.conv1_kernel_size,
      D=3,
      config=config)

    # teacher = ResUNetBN5C(
    #   num_feats,
    #   config.model_n_out,
    #   bn_momentum=config.bn_momentum,
    #   normalize_feature=config.normalize_feature,
    #   conv1_kernel_size=config.conv1_kernel_size,
    #   D=3,
    #   config=config)

    checkpoint = torch.load(config.teacher_weight)
    # teacher.load_state_dict(checkpoint['state_dict'])


    miss = student.load_state_dict(checkpoint['state_dict'], strict=False)
    print('miss part:', miss)

    self.config = config
    # self.teacher = teacher
    self.student = student

    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf
    if (self.best_val_metric == "rre" or self.best_val_metric == "rte"):
      self.best_val = np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.optimizer = getattr(optim, config.optimizer)(
      self.student.model_para.parameters(),
      lr=config.lr,
      momentum=config.momentum,
      weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
      config,
      open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
      indent=4,
      sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    # self.test_valid = True if self.val_data_loader is not None else False
    self.test_valid = config.test_valid
    self.log_step = int(np.sqrt(self.config.batch_size))
    # self.teacher = self.teacher.to(self.device)
    self.student = self.student.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    # if config.resume is not None:
    #   if osp.isfile(config.resume):
    #     logging.info("=> loading checkpoint '{}'".format(config.resume))
    #     state = torch.load(config.resume)
    #     self.start_epoch = state['epoch']
    #     model.load_state_dict(state['state_dict'])
    #     self.scheduler.load_state_dict(state['scheduler'])
    #     self.optimizer.load_state_dict(state['optimizer'])
    #
    #     if 'best_val' in state.keys():
    #       self.best_val = state['best_val']
    #       self.best_val_epoch = state['best_val_epoch']
    #       self.best_val_metric = state['best_val_metric']
    #   else:
    #     raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight
    self.w_kd = config.w_kd
    self.kd_loss = CriterionKD(temperature=config.temperature, use_softmax=config.use_softmax)

    # self.kp_list = ['points', 'neighbors', 'pools', 'upsamples', 'features', 'stack_lengths']
    # ---- train start ----

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()
          self._save_checkpoint(
            epoch=epoch,
            val_dict=val_dict,
            best_val_metric=self.best_val_metric
          )

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if (self.best_val_metric == "feat_match_ratio" or self.best_val_metric == "success"):
          if self.best_val < val_dict[self.best_val_metric]:
            logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )
        elif (self.best_val_metric == "rre" or self.best_val_metric == "rte"):
          if self.best_val > val_dict[self.best_val_metric]:
            logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )

  def _save_checkpoint(self, epoch, val_dict, best_val_metric, filename='checkpoint'):
    state = {
      'epoch': epoch,
      'state_dict': self.student.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'scheduler': self.scheduler.state_dict(),
      'config': self.config,
      'best_val': self.best_val,
      'best_val_epoch': self.best_val_epoch,
      'best_val_metric': self.best_val_metric
    }
    content = f"{best_val_metric}_{val_dict[best_val_metric]}"
    filename = os.path.join(self.checkpoint_dir, f'{filename}_epoch_{epoch}_{content}.pth')

    logging.info(f"Saving checkpoint: {filename} ...")
    torch.save(state, filename)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
      np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
      np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
      np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  # ---- current train ----
  def _train_epoch(self, epoch):
    gc.collect()
    self.student.train()
    # self.teacher.eval()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss_kd, batch_loss = 0, 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        # image0 = input_dict['image0'].to(self.device)
        # image1 = input_dict['image1'].to(self.device)

        batch0 = input_dict['batch0']
        for k, v in batch0.items():
          if type(v) == list:
            batch0[k] = [item.to(self.device) for item in v]
          elif type(v) == dict:
            pass
          else:
            batch0[k] = v.to(self.device)
        batch1 = input_dict['batch1']
        for k, v in batch1.items():
          if type(v) == list:
            batch1[k] = [item.to(self.device) for item in v]
          elif type(v) == dict:
            pass
          else:
            batch1[k] = v.to(self.device)

        sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'].to(self.device),
          coordinates=input_dict['sinput0_C'].to(self.device))
        F0, _ = self.student(sinput0, batch0)
        # _, F0_median_t = self.teacher(sinput0, image0)

        sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'].to(self.device),
          coordinates=input_dict['sinput1_C'].to(self.device))
        F1, _ = self.student(sinput1, batch1)
        # _, F1_median_t = self.teacher(sinput1, image1)

        pos_pairs = input_dict['correspondences']
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
          F0,
          F1,
          pos_pairs,
          num_pos=self.config.num_pos_per_batch * self.config.batch_size,
          num_hn_samples=self.config.num_hn_samples_per_batch *
                         self.config.batch_size)

        pos_loss /= iter_size
        neg_loss /= iter_size

        loss = pos_loss + self.neg_weight * neg_loss
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/distillation_loss', batch_loss_kd, start_iter + curr_iter)
        logging.info(
          "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
          .format(epoch, curr_iter,
                  len(self.data_loader) //
                  iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
          "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
            data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  # ---- current valid ----
  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.student.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, success_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      # image0 = input_dict['image0'].to(self.device)
      # sinput0 = ME.SparseTensor(
      #   input_dict['sinput0_F'].to(self.device),
      #   coordinates=input_dict['sinput0_C'].to(self.device))
      # F0 = self.model(sinput0, image0).F
      #
      # image1 = input_dict['image1'].to(self.device)
      # sinput1 = ME.SparseTensor(
      #   input_dict['sinput1_F'].to(self.device),
      #   coordinates=input_dict['sinput1_C'].to(self.device))
      # F1 = self.model(sinput1, image1).F
      batch0 = input_dict['batch0']
      for k, v in batch0.items():
        if type(v) == list:
          batch0[k] = [item.to(self.device) for item in v]
        elif type(v) == dict:
          pass
        else:
          batch0[k] = v.to(self.device)
      batch1 = input_dict['batch1']
      for k, v in batch1.items():
        if type(v) == list:
          batch1[k] = [item.to(self.device) for item in v]
        elif type(v) == dict:
          pass
        else:
          batch1[k] = v.to(self.device)

      sinput0 = ME.SparseTensor(
        input_dict['sinput0_F'].to(self.device),
        coordinates=input_dict['sinput0_C'].to(self.device))
      F0, _ = self.student(sinput0, batch0)

      sinput1 = ME.SparseTensor(
        input_dict['sinput1_F'].to(self.device),
        coordinates=input_dict['sinput1_C'].to(self.device))
      F1, _ = self.student(sinput1, batch1)
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
      if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
        success_meter.update(1)
      else:
        success_meter.update(0)

      hit_ratio = self.evaluate_hit_ratio(
        xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
          f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
          f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
          f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f}, Success:{success_meter.avg:.3f}",
          f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
        ]))
        data_timer.reset()

    logging.info(' '.join([
      f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
      f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
      "loss": loss_meter.avg,
      "rre": rre_meter.avg,
      "rte": rte_meter.avg,
      'feat_match_ratio': feat_match_ratio.avg,
      'hit_ratio': hit_ratio_meter.avg,
      "success": success_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1) ** 2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()

  def distillation_loss(self, F0_median_s, F0_median_t, F1_median_s, F1_median_t):
    loss0 = self.kd_loss(F0_median_s, F0_median_t)
    loss1 = self.kd_loss(F1_median_s, F1_median_t)
    return loss0 + loss1


#only frozen encoder
class Distillation_frozen_encoder:
  def __init__(self, config, data_loader, val_data_loader):

    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    # student use a kpfcn to learn semantic feature from teacher's resnet34
    student = ResUNetBN4C(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      normalize_feature=config.normalize_feature,
      conv1_kernel_size=config.conv1_kernel_size,
      D=3,
      config=config)

    teacher = ResUNetBN5C(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      normalize_feature=config.normalize_feature,
      conv1_kernel_size=config.conv1_kernel_size,
      D=3,
      config=config)

    checkpoint = torch.load(config.teacher_weight)
    teacher.load_state_dict(checkpoint['state_dict'])


    miss = student.load_state_dict(checkpoint['state_dict'], strict=False)
    print('miss part:', miss)

    self.config = config
    self.teacher = teacher
    self.student = student

    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf
    if (self.best_val_metric == "rre" or self.best_val_metric == "rte"):
      self.best_val = np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for param in self.student.conv1.parameters():
      param.requires_grad = False
    for param in self.student.norm1.parameters():
      param.requires_grad = False
    for param in self.student.block1.parameters():
      param.requires_grad = False
    for param in self.student.conv2.parameters():
      param.requires_grad = False
    for param in self.student.norm2.parameters():
      param.requires_grad = False
    for param in self.student.block2.parameters():
      param.requires_grad = False
    for param in self.student.conv3.parameters():
      param.requires_grad = False
    for param in self.student.norm3.parameters():
      param.requires_grad = False
    for param in self.student.block3.parameters():
      param.requires_grad = False
    for param in self.student.conv4.parameters():
      param.requires_grad = False
    for param in self.student.norm4.parameters():
      param.requires_grad = False
    for param in self.student.block4.parameters():
      param.requires_grad = False

    self.optimizer = getattr(optim, config.optimizer)(
      filter(lambda p: p.requires_grad, self.student.parameters()),
      lr=config.lr,
      momentum=config.momentum,
      weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
      config,
      open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
      indent=4,
      sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    # self.test_valid = True if self.val_data_loader is not None else False
    self.test_valid = config.test_valid
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.teacher = self.teacher.to(self.device)
    self.student = self.student.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    # if config.resume is not None:
    #   if osp.isfile(config.resume):
    #     logging.info("=> loading checkpoint '{}'".format(config.resume))
    #     state = torch.load(config.resume)
    #     self.start_epoch = state['epoch']
    #     model.load_state_dict(state['state_dict'])
    #     self.scheduler.load_state_dict(state['scheduler'])
    #     self.optimizer.load_state_dict(state['optimizer'])
    #
    #     if 'best_val' in state.keys():
    #       self.best_val = state['best_val']
    #       self.best_val_epoch = state['best_val_epoch']
    #       self.best_val_metric = state['best_val_metric']
    #   else:
    #     raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight
    self.w_kd = config.w_kd
    self.kd_loss = CriterionKD(temperature=config.temperature, use_softmax=config.use_softmax)

    # self.kp_list = ['points', 'neighbors', 'pools', 'upsamples', 'features', 'stack_lengths']
    # ---- train start ----

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()
          self._save_checkpoint(
            epoch=epoch,
            val_dict=val_dict,
            best_val_metric=self.best_val_metric
          )

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if (self.best_val_metric == "feat_match_ratio" or self.best_val_metric == "success"):
          if self.best_val < val_dict[self.best_val_metric]:
            logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )
        elif (self.best_val_metric == "rre" or self.best_val_metric == "rte"):
          if self.best_val > val_dict[self.best_val_metric]:
            logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
            )
            self.best_val = val_dict[self.best_val_metric]
            self.best_val_epoch = epoch
            self._save_checkpoint(
              epoch=epoch,
              filename='best_val_checkpoint',
              val_dict=val_dict,
              best_val_metric=self.best_val_metric
            )
          else:
            logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
            )

  def _save_checkpoint(self, epoch, val_dict, best_val_metric, filename='checkpoint'):
    state = {
      'epoch': epoch,
      'state_dict': self.student.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'scheduler': self.scheduler.state_dict(),
      'config': self.config,
      'best_val': self.best_val,
      'best_val_epoch': self.best_val_epoch,
      'best_val_metric': self.best_val_metric
    }
    content = f"{best_val_metric}_{val_dict[best_val_metric]}"
    filename = os.path.join(self.checkpoint_dir, f'{filename}_epoch_{epoch}_{content}.pth')

    logging.info(f"Saving checkpoint: {filename} ...")
    torch.save(state, filename)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
      np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
      np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
      np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  # ---- current train ----
  def _train_epoch(self, epoch):
    gc.collect()
    self.student.train()
    self.teacher.eval()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss_kd, batch_loss = 0, 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)

        image0 = input_dict['image0'].to(self.device)
        image1 = input_dict['image1'].to(self.device)

        batch0 = input_dict['batch0']
        for k, v in batch0.items():
          if type(v) == list:
            batch0[k] = [item.to(self.device) for item in v]
          elif type(v) == dict:
            pass
          else:
            batch0[k] = v.to(self.device)
        batch1 = input_dict['batch1']
        for k, v in batch1.items():
          if type(v) == list:
            batch1[k] = [item.to(self.device) for item in v]
          elif type(v) == dict:
            pass
          else:
            batch1[k] = v.to(self.device)

        sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'].to(self.device),
          coordinates=input_dict['sinput0_C'].to(self.device))
        F0, F0_median_s = self.student(sinput0, batch0)
        # _, F0_median_t = self.teacher(sinput0, image0)

        sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'].to(self.device),
          coordinates=input_dict['sinput1_C'].to(self.device))
        F1, F1_median_s = self.student(sinput1, batch1)
        # _, F1_median_t = self.teacher(sinput1, image1)

        pos_pairs = input_dict['correspondences']
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
          F0,
          F1,
          pos_pairs,
          num_pos=self.config.num_pos_per_batch * self.config.batch_size,
          num_hn_samples=self.config.num_hn_samples_per_batch *
                         self.config.batch_size)

        pos_loss /= iter_size
        neg_loss /= iter_size

        # loss_kd = self.distillation_loss(F0_median_s, F0_median_t, F1_median_s, F1_median_t)
        # loss_kd /= iter_size

        loss = pos_loss + self.neg_weight * neg_loss   #+ self.w_kd * loss_kd
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()
        # batch_loss_kd += loss_kd.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/distillation_loss', batch_loss_kd, start_iter + curr_iter)
        logging.info(
          "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
          .format(epoch, curr_iter,
                  len(self.data_loader) //
                  iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
          "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
            data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  # ---- current valid ----
  def _valid_epoch(self):
    # Change the network to evaluation mode
    self.student.eval()
    self.val_data_loader.dataset.reset_seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter, success_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      # image0 = input_dict['image0'].to(self.device)
      # sinput0 = ME.SparseTensor(
      #   input_dict['sinput0_F'].to(self.device),
      #   coordinates=input_dict['sinput0_C'].to(self.device))
      # F0 = self.model(sinput0, image0).F
      #
      # image1 = input_dict['image1'].to(self.device)
      # sinput1 = ME.SparseTensor(
      #   input_dict['sinput1_F'].to(self.device),
      #   coordinates=input_dict['sinput1_C'].to(self.device))
      # F1 = self.model(sinput1, image1).F
      batch0 = input_dict['batch0']
      for k, v in batch0.items():
        if type(v) == list:
          batch0[k] = [item.to(self.device) for item in v]
        elif type(v) == dict:
          pass
        else:
          batch0[k] = v.to(self.device)
      batch1 = input_dict['batch1']
      for k, v in batch1.items():
        if type(v) == list:
          batch1[k] = [item.to(self.device) for item in v]
        elif type(v) == dict:
          pass
        else:
          batch1[k] = v.to(self.device)

      sinput0 = ME.SparseTensor(
        input_dict['sinput0_F'].to(self.device),
        coordinates=input_dict['sinput0_C'].to(self.device))
      F0, _ = self.student(sinput0, batch0)

      sinput1 = ME.SparseTensor(
        input_dict['sinput1_F'].to(self.device),
        coordinates=input_dict['sinput1_C'].to(self.device))
      F1, _ = self.student(sinput1, batch1)
      feat_timer.toc()

      matching_timer.tic()
      xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
      if not np.isnan(rre):
        rre_meter.update(rre)

      # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
      if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
        success_meter.update(1)
      else:
        success_meter.update(0)

      hit_ratio = self.evaluate_hit_ratio(
        xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)
      matching_timer.toc()

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        logging.info(' '.join([
          f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
          f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
          f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f}, Success:{success_meter.avg:.3f}",
          f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
        ]))
        data_timer.reset()

    logging.info(' '.join([
      f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
      f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
      "loss": loss_meter.avg,
      "rre": rre_meter.avg,
      "rte": rte_meter.avg,
      'feat_match_ratio': feat_match_ratio.avg,
      'hit_ratio': hit_ratio_meter.avg,
      "success": success_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
      N0 = min(len(F0), subsample_size)
      N1 = min(len(F1), subsample_size)
      inds0 = np.random.choice(len(F0), N0, replace=False)
      inds1 = np.random.choice(len(F1), N1, replace=False)
      F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
    if subsample_size > 0 and subsample:
      return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
      return xyz0, xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1) ** 2).sum(1) + 1e-6)
    return (dist < thresh).float().mean().item()

  def distillation_loss(self, F0_median_s, F0_median_t, F1_median_s, F1_median_t):
    loss0 = self.kd_loss(F0_median_s, F0_median_t)
    loss1 = self.kd_loss(F1_median_s, F1_median_t)
    return loss0 + loss1
