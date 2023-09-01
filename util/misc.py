import torch
import numpy as np
import MinkowskiEngine as ME
import time
from lib.kpsampling import batch_neighbors_kpconv, batch_grid_subsampling_kpconv


def generate_data_for_kpfcn(src_pcd, config):
  batched_points0_list = []
  batched_features0_list = []
  batched_lengths0_list = []

  batched_points0_list.append(src_pcd)
  batched_lengths0_list.append(len(src_pcd))

  batched_points0 = torch.from_numpy(np.concatenate(batched_points0_list, axis=0))
  batched_features0 = torch.ones([len(batched_points0), 1])
  batched_lengths0 = torch.from_numpy(np.array(batched_lengths0_list)).int()

  # Starting radius of convolutions
  r_normal = config.first_subsampling_dl * config.conv_radius

  # Starting layer
  layer_blocks = []
  layer = 0

  # Lists of inputs
  input_points0 = []
  input_neighbors0 = []
  input_pools0 = []
  input_batches_len0 = []

  for block_i, block in enumerate(config.architecture):

    # Stop when meeting a global pooling or upsampling
    if 'global' in block or 'upsample' in block:
      break

    # Get all blocks of the layer
    if not ('pool' in block or 'strided' in block):
      layer_blocks += [block]
      if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
        continue

    # Convolution neighbors indices
    # *****************************

    if layer_blocks:
      # Convolutions are done in this layer, compute the neighbors with the good radius
      if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
        r = r_normal * config.deform_radius / config.conv_radius
      else:
        r = r_normal
      conv_i0 = batch_neighbors_kpconv(batched_points0, batched_points0, batched_lengths0, batched_lengths0, r,
                                       config.neighborhood_limits[layer])

    else:
      # This layer only perform pooling, no neighbors required
      conv_i0 = torch.zeros((0, 1), dtype=torch.int64)

    # Pooling neighbors indices
    # *************************

    # If end of layer is a pooling operation
    if 'pool' in block or 'strided' in block:

      # New subsampling length
      dl = 2 * r_normal / config.conv_radius

      # Subsampled points
      pool_p0, pool_b0 = batch_grid_subsampling_kpconv(batched_points0, batched_lengths0, sampleDl=dl)

      # Radius of pooled neighbors
      if 'deformable' in block:
        r = r_normal * config.deform_radius / config.conv_radius
      else:
        r = r_normal

      # Subsample indices
      pool_i0 = batch_neighbors_kpconv(pool_p0, batched_points0, pool_b0, batched_lengths0, r,
                                       config.neighborhood_limits[layer])



    else:
      # No pooling in the end of this layer, no pooling indices required
      pool_i0 = torch.zeros((0, 1), dtype=torch.int64)
      pool_p0 = torch.zeros((0, 3), dtype=torch.float32)
      pool_b0 = torch.zeros((0,), dtype=torch.int64)


    # Updating input lists
    input_points0 += [batched_points0.float()]
    input_neighbors0 += [conv_i0.long()]
    input_pools0 += [pool_i0.long()]
    input_batches_len0 += [batched_lengths0]

    # New points for next layer
    batched_points0 = pool_p0
    batched_lengths0 = pool_b0

    # Update radius and reset blocks
    r_normal *= 2
    layer += 1
    layer_blocks = []

  batch0 = {
    'points': input_points0,
    'neighbors': input_neighbors0,
    'pools': input_pools0,
    # 'upsamples': input_upsamples,
    'features': batched_features0.float(),
    'stack_lengths': input_batches_len0,
  }
  return batch0



def _hash(arr, M):
  if isinstance(arr, np.ndarray):
    N, D = arr.shape
  else:
    N, D = len(arr[0]), len(arr)

  hash_vec = np.zeros(N, dtype=np.int64)
  for d in range(D):
    if isinstance(arr, np.ndarray):
      hash_vec += arr[:, d] * M**d
    else:
      hash_vec += arr[d] * M**d
  return hash_vec


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True,
                     image=None
                     ):
  '''
  xyz is a N x 3 matrix
  rgb is a N x 3 matrix and all color must range from [0, 1] or None
  normal is a N x 3 matrix and all normal range from [-1, 1] or None

  if both rgb and normal are None, we use Nx1 one vector as an input

  if device is None, it tries to use gpu by default

  if skip_check is True, skip rigorous checks to speed up

  model = model.to(device)
  xyz, feats = extract_features(model, xyz)
  '''
  if is_eval:
    model.eval()

  if not skip_check:
    assert xyz.shape[1] == 3

    N = xyz.shape[0]
    if rgb is not None:
      assert N == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  # Voxelize xyz and feats
  coords = np.floor(xyz / voxel_size)
  coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
  # coords,inds = coords[:5000],inds[:5000]
  # Convert to batched coords compatible with ME
  coords = ME.utils.batched_coordinates([coords])
  return_coords = xyz[inds]

  feats = feats[inds]

  feats = torch.tensor(feats, dtype=torch.float32)
  coords = torch.tensor(coords, dtype=torch.int32)


  stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

  image = torch.as_tensor(image,dtype=torch.float32,device=device)

  # start = time.time()
  F = model(stensor,image).F
  # end = time.time()
  # t = end - start

  return return_coords,F


def extract_features_kp(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True,
                     config=None
                     ):


  if is_eval:
    model.eval()

  if not skip_check:
    assert xyz.shape[1] == 3

    N = xyz.shape[0]
    if rgb is not None:
      assert N == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  # Voxelize xyz and feats
  coords = np.floor(xyz / voxel_size)
  coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
  # coords,inds = coords[:5000],inds[:5000]
  # Convert to batched coords compatible with ME
  coords = ME.utils.batched_coordinates([coords])
  return_coords = xyz[inds]

  feats = feats[inds]

  feats = torch.tensor(feats, dtype=torch.float32)
  coords = torch.tensor(coords, dtype=torch.int32)

  # sample pcd and idx for kpfcn
  batch = generate_data_for_kpfcn(return_coords, config)
  for k, v in batch.items():
    if type(v) == list:
      batch[k] = [item.to(device) for item in v]
    elif type(v) == dict:
      pass
    else:
      batch[k] = v.to(device)

  stensor = ME.SparseTensor(feats, coordinates=coords, device=device)


  # start = time.time()
  F, _ = model(stensor,batch)
  # end = time.time()
  # t = end - start

  return return_coords,F