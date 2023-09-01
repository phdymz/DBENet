import open3d as o3d
import numpy as np
from functools import partial
import torch
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from config.kpfcn_config import get_kp_config
from lib.data_loaders import ThreeDMatchPairDataset, t
from config_3dmatch import get_config as get_config_3dmatch
import MinkowskiEngine as ME


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0,
                                  random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(
            s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_descriptor(list_data, config, neighborhood_limits):
    xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans, p_image, q_image = list(
        zip(*list_data))
    xyz_batch0, xyz_batch1 = [], []
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    p_image_batch, q_image_batch = [], []

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]

        xyz_batch0.append(to_tensor(xyz0[batch_id]))
        xyz_batch1.append(to_tensor(xyz1[batch_id]))

        trans_batch.append(to_tensor(trans[batch_id]))

        matching_inds_batch.append(
            torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
        len_batch.append([N0, N1])

        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

        # image batch
        p_image_batch.append(to_tensor(p_image[batch_id][None, :, :, :]))
        q_image_batch.append(to_tensor(q_image[batch_id][None, :, :, :]))

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
    p_image_batch = torch.cat(p_image_batch, 0).float()
    q_image_batch = torch.cat(q_image_batch, 0).float()


    #KPConv
    batched_points0_list = []
    batched_points1_list = []
    batched_features0_list = []
    batched_features1_list = []
    batched_lengths0_list = []
    batched_lengths1_list = []


    for ind, (
    src_pcd, tgt_pcd, _, _, _, _, _, _, _, _) in enumerate(
            list_data):
        batched_points0_list.append(src_pcd)
        batched_points1_list.append(tgt_pcd)
        batched_lengths0_list.append(len(src_pcd))
        batched_lengths1_list.append(len(tgt_pcd))



    batched_points0 = torch.from_numpy(np.concatenate(batched_points0_list, axis=0))
    batched_points1 = torch.from_numpy(np.concatenate(batched_points1_list, axis=0))
    batched_features0 = torch.ones([len(batched_points0), 1])
    batched_features1 = torch.ones([len(batched_points1), 1])
    batched_lengths0 = torch.from_numpy(np.array(batched_lengths0_list)).int()
    batched_lengths1 = torch.from_numpy(np.array(batched_lengths1_list)).int()

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points0 = []
    input_points1 = []
    input_neighbors0 = []
    input_neighbors1 = []
    input_pools0 = []
    input_pools1 = []
    # input_upsamples0 = []
    # input_upsamples1 = []
    input_batches_len0 = []
    input_batches_len1 = []


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
                                            neighborhood_limits[layer])
            conv_i1 = batch_neighbors_kpconv(batched_points1, batched_points1, batched_lengths1, batched_lengths1, r,
                                             neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i0 = torch.zeros((0, 1), dtype=torch.int64)
            conv_i1 = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p0, pool_b0 = batch_grid_subsampling_kpconv(batched_points0, batched_lengths0, sampleDl=dl)
            pool_p1, pool_b1 = batch_grid_subsampling_kpconv(batched_points1, batched_lengths1, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i0 = batch_neighbors_kpconv(pool_p0, batched_points0, pool_b0, batched_lengths0, r,
                                            neighborhood_limits[layer])
            pool_i1 = batch_neighbors_kpconv(pool_p1, batched_points1, pool_b1, batched_lengths1, r,
                                             neighborhood_limits[layer])


            # # Upsample indices (with the radius of the next layer to keep wanted density)
            # up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
            #                               neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i0 = torch.zeros((0, 1), dtype=torch.int64)
            pool_i1 = torch.zeros((0, 1), dtype=torch.int64)
            pool_p0 = torch.zeros((0, 3), dtype=torch.float32)
            pool_p1 = torch.zeros((0, 3), dtype=torch.float32)
            pool_b0 = torch.zeros((0,), dtype=torch.int64)
            pool_b1 = torch.zeros((0,), dtype=torch.int64)
            # up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points0 += [batched_points0.float()]
        input_points1 += [batched_points1.float()]
        input_neighbors0 += [conv_i0.long()]
        input_neighbors1 += [conv_i1.long()]
        input_pools0 += [pool_i0.long()]
        input_pools1 += [pool_i1.long()]
        # input_upsamples += [up_i.long()]
        input_batches_len0 += [batched_lengths0]
        input_batches_len1 += [batched_lengths1]

        # New points for next layer
        batched_points0 = pool_p0
        batched_points1 = pool_p1
        batched_lengths0 = pool_b0
        batched_lengths1 = pool_b1

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
    batch1 = {
            'points': input_points1,
            'neighbors': input_neighbors1,
            'pools': input_pools1,
            # 'upsamples': input_upsamples,
            'features': batched_features1.float(),
            'stack_lengths': input_batches_len1,
    }

        # 'sample': sample
    return {
        'pcd0': xyz_batch0,
        'pcd1': xyz_batch1,
        'image0': p_image_batch,
        'image1': q_image_batch,
        'sinput0_C': coords_batch0,
        'sinput0_F': feats_batch0.float(),
        'sinput1_C': coords_batch1,
        'sinput1_F': feats_batch1.float(),
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch,
        #for kpconv
        'batch0': batch0,
        'batch1': batch1
    }


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        # counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in
                  batched_input['batch0']['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)


        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits



def get_dataloader(dataset, batch_size=1, num_workers=4, shuffle=True, neighborhood_limits=None):
    config = dataset.config
    if neighborhood_limits is None:
        # neighborhood_limits = calibrate_neighbors(dataset, dataset.config, collate_fn=collate_fn_descriptor)
        neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn=collate_fn_descriptor)
    print("neighborhood:", neighborhood_limits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
        collate_fn=partial(collate_fn_descriptor, config=dataset.config, neighborhood_limits=neighborhood_limits),
        drop_last=False
    )
    return dataloader, neighborhood_limits




if __name__ == "__main__":
    kp_config = get_kp_config()
    # dconfig = vars(config)
    print('load config for kpconv')

    config = get_config_3dmatch()
