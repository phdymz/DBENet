from model.blocks import *
import torch.nn.functional as F
import numpy as np
from torch import nn as nn
from config_3dmatch import get_config


class KPFCN(nn.Module):

    def __init__(self, config):
        super(KPFCN, self).__init__()

        ############
        # Parameters
        ############
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim

        #####################
        # List Encoder blocks
        #####################
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        # #####################
        # # List Decoder blocks
        # #####################
        #
        # # Save all block operations in a list of modules
        # self.decoder_blocks = nn.ModuleList()
        # self.decoder_concats = []
        #
        # # Find first upsampling block
        # start_i = 0
        # for block_i, block in enumerate(config.architecture):
        #     if 'upsample' in block:
        #         start_i = block_i
        #         break
        #
        # # Loop over consecutive blocks
        # for block_i, block in enumerate(config.architecture[start_i:]):
        #
        #     # Add dimension of skip connection concat
        #     if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
        #         in_dim += self.encoder_skip_dims[layer]
        #         self.decoder_concats.append(block_i)
        #
        #     # Apply the good block function defining tf ops
        #     self.decoder_blocks.append(block_decider(block,
        #                                             r,
        #                                             in_dim,
        #                                             out_dim,
        #                                             layer,
        #                                             config))
        #
        #     # Update dimension of input from output
        #     in_dim = out_dim
        #
        #     # Detect change to a subsampled layer
        #     if 'upsample' in block:
        #         # Update radius and feature dimension for next layer
        #         layer -= 1
        #         r *= 0.5
        #         out_dim = out_dim // 2


    def forward(self, batch):
        # Get input features

        x = batch['features'].clone().detach()
        # 1. joint encoder part
        self.skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                self.skip_x.append(x)
            x = block_op(x, batch)  # [N,C]

        # for block_i, block_op in enumerate(self.decoder_blocks):
        #     if block_i in self.decoder_concats:
        #         x = torch.cat([x, self.skip_x.pop()], dim=1)
        #     x = block_op(x, batch)

        # features = F.normalize(x, p=2, dim=-1)
        return x


if __name__ == "__main__":
    config = get_config()
    model = KPFCN(config)
    print(model)