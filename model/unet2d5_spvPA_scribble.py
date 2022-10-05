# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F

import numpy as np
import cv2

from model.params.networks.blocks.convolutions import Convolution, ResidualUnit
from model.params.networks.blocks.attentionblock import AttentionBlock1, AttentionBlock2

from monai.networks.layers.factories import Norm, Act
from monai.networks.layers.simplelayers import SkipConnection

class UNet2d5_spvPA_scribble(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_sizes,
        sample_kernel_sizes,
        num_res_units=0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
        attention_module=True,
        channel=32
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == (len(strides)) + 1 == len(sample_kernel_sizes) + 1
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.sample_kernel_sizes = sample_kernel_sizes
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.attention_module = attention_module
        self.att_maps = []
        self.fea_maps = []
        self.edge_module = Edge_Module()
        self.fuse_canny_edge = Convolution(
            dimensions=dimensions,
            in_channels=2,
            out_channels=1,
            strides=1,
            kernel_size=(1,1,1),
            act=None,
            norm=None,
            bias=False
        )
        self.aspp = _AtrousSpatialPyramidPoolingModule(160, 32, output_stride=16)
        self.after_aspp_conv5 = Convolution(
            dimensions=dimensions,
            in_channels=channel*6,
            out_channels=channel,
            strides=1,
            kernel_size=(1,1,1),
            act=None,
            norm=None,
            bias=False
        )
        self.after_aspp_conv2 = Convolution(
            dimensions=dimensions,
            in_channels=64,
            out_channels=channel,
            strides=1,
            kernel_size=(1,1,1),
            act=None,
            norm=None,
            bias=False
        )
        self.final_sal_seg = nn.Sequential(
            Convolution(
                dimensions=dimensions,
                in_channels=channel*2,
                out_channels=channel,
                strides=1,
                kernel_size=(3,3,3),
                act=Act.RELU,
                norm=None,
                bias=False
            ),
            Convolution(
                dimensions=dimensions,
                in_channels=channel,
                out_channels=channel,
                strides=1,
                kernel_size=(3,3,3),
                act=Act.RELU,
                norm=None,
                bias=False
            ),
            Convolution(
                dimensions=dimensions,
                in_channels=channel,
                out_channels=out_channels,
                strides=1,
                kernel_size=(1,1,1),
                act=None,
                norm=None,
                bias=False
            )
        )

        self.sal_conv = Convolution(
            dimensions=dimensions,
            in_channels=out_channels,
            out_channels=channel,
            strides=1,
            kernel_size=(3,3,3),
            act=None,
            norm=None,
            bias=False
        )
        self.edge_conv = Convolution(
            dimensions=dimensions,
            in_channels=1,
            out_channels=channel,
            strides=1,
            kernel_size=(3,3,3),
            act=None,
            norm=None,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.rcab_sal_edge = RCAB(channel*2)
        self.fused_edge_sal = Convolution(
            dimensions=dimensions,
            in_channels=64,
            out_channels=out_channels,
            strides=1,
            kernel_size=(3,3,3),
            act=None,
            norm=None,
            bias=False
        )
        self.init_to_one_channel = Convolution(
            dimensions=dimensions,
            in_channels=out_channels,
            out_channels=1,
            strides=1,
            kernel_size=(1,1,1),
            act=None,
            norm=None,
            bias=False
        )
        self.ref_to_one_channel = Convolution(
            dimensions=dimensions,
            in_channels=out_channels,
            out_channels=1,
            strides=1,
            kernel_size=(1,1,1),
            act=None,
            norm=None,
            bias=False
        )

        def _create_block(inc, outc, channels, strides, kernel_sizes, sample_kernel_sizes, is_top):
            """
            Builds the UNet2d5_spvPA structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]
            k = kernel_sizes[0]
            sk = sample_kernel_sizes[0]

            # create layer in downsampling path
            down = self._get_down_layer(in_channels=inc, out_channels=c, kernel_size=k)
            # print("down")
            downsample = self._get_downsample_layer(in_channels=c, out_channels=c, strides=s, kernel_size=sk)
            # print("downsample")
            # print("len(channels)", len(channels))

            if len(channels) > 2:
                # continue recursion down
                subblock = _create_block(
                    c, channels[1], channels[1:], strides[1:], kernel_sizes[1:], sample_kernel_sizes[1:], is_top=False
                )
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(
                    in_channels=c,
                    out_channels=channels[1],
                    kernel_size=kernel_sizes[1],
                )

            upsample = self._get_upsample_layer(in_channels=channels[1], out_channels=c, strides=s, up_kernel_size=sk)
            # print("upsample")
            subblock_with_resampling = nn.Sequential(downsample, subblock, upsample)
            # print("subblock_with_resampling")

            # create layer in upsampling path
            up = self._get_up_layer(in_channels=2 * c, out_channels=outc, kernel_size=k, is_top=is_top)
            # print("up")
            # import pdb;pdb.set_trace()

            return nn.Sequential(down, SkipConnection(subblock_with_resampling), up)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, self.kernel_sizes, self.sample_kernel_sizes, True
        )

        # register forward hooks on all Attentionblock1 modules, to save the attention maps
        if self.attention_module:
            for layer in self.model.modules():
                if type(layer) == AttentionBlock1:
                    layer.register_forward_hook(self.hook_save_attention_map)

    def hook_save_attention_map(self, module, inp, outp):
        if len(self.att_maps) == len(self.channels):
            self.att_maps = []
            self.fea_maps = []
        self.att_maps.append(outp[0])  # get first element of output (Attentionblock1 returns (att, x) )
        self.fea_maps.append(outp[1])


    def _get_att_layer(self, in_channels, out_channels, kernel_size):
        att1 = AttentionBlock1(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        att2 = AttentionBlock2(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        return nn.Sequential(att1, att2)

    def _get_down_layer(self, in_channels, out_channels, kernel_size):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )

    def _get_downsample_layer(self, in_channels, out_channels, strides, kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=False,
        )
        return conv

    def _get_bottom_layer(self, in_channels, out_channels, kernel_size):
        conv = self._get_down_layer(in_channels, out_channels, kernel_size)
        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)
            return nn.Sequential(att_layer, conv)
        else:
            return conv

    def _get_upsample_layer(self, in_channels, out_channels, strides, up_kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=True,
        )
        return conv

    def _get_up_layer(self, in_channels, out_channels, kernel_size, is_top):

        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=1,  # why not self.num_res_units?
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )

        if self.attention_module and self.num_res_units > 0:
            return nn.Sequential(att_layer, ru)
        elif self.attention_module and not self.num_res_units > 0:
            return att_layer
        elif self.num_res_units > 0 and not self.attention_module:
            return ru
        elif not self.attention_module and not self.num_res_units > 0:
            return nn.Identity
        else:
            raise NotImplementedError


    def forward(self, x):
        x_size = x.size()
        x1 = self.model(x)

        edge_map = self.edge_module(self.fea_maps)
        edge_out = torch.sigmoid(edge_map)
        
        ####
        canny = torch.zeros(x_size)
        for d in range(x_size[-1]):
            x_d = (x[:,:,:,:,d] - x[:,:,:,:,d].min()) / (x[:,:,:,:,d].max() - x[:,:,:,:,d].min()) * 255.0
            im_arr = x_d.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8) # BHWC
            im_arr = np.concatenate((im_arr, im_arr, im_arr), axis=-1)
            canny_d = np.zeros((x_size[0], 1, x_size[2], x_size[3])) # B*1*H*W
            for i in range(x_size[0]):
                canny_d[i] = cv2.Canny(im_arr[i], 10, 100)
            canny[:,:,:,:,d] = torch.from_numpy(canny_d)
        canny = (canny / 255.0).cuda().float()
        # import pdb;pdb.set_trace()
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.fuse_canny_edge(cat)
        acts = torch.sigmoid(acts)

        x5 = self.fea_maps[1]
        x2 = self.fea_maps[4]
        x5 = self.aspp(x5, acts) # torch.Size([2, 192, 14, 14, 8])
        x_conv5 = self.after_aspp_conv5(x5) # torch.Size([2, 32, 14, 14, 8])
        x_conv2 = self.after_aspp_conv2(x2) # torch.Size([2, 32, 112, 112, 32])

        x_conv5_up = F.interpolate(x_conv5, x2.size()[2:], mode='trilinear', align_corners=True)
        feat_fuse = torch.cat([x_conv5_up, x_conv2], 1)

        sal_init = self.final_sal_seg(feat_fuse)
        sal_init = F.interpolate(sal_init, x_size[2:], mode='trilinear')

        

        sal_feature = self.sal_conv(sal_init)
        edge_feature = self.edge_conv(edge_map)
        sal_edge_feature = self.relu(torch.cat((sal_feature, edge_feature), 1))
        sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
        sal_ref = self.fused_edge_sal(sal_edge_feature)
        sal_init_one_channel = self.init_to_one_channel(sal_init)
        sal_ref_one_channel = self.ref_to_one_channel(sal_ref)
        # import pdb;pdb.set_trace()

        return x1, self.att_maps, sal_init, edge_map, sal_ref, sal_init_one_channel, sal_ref_one_channel


class Edge_Module(nn.Module):

    def __init__(
        self, 
        dimensions=3,
        in_channels=[32, 96, 128], 
        mid_channels=16,
        kernel_sizes = (
            (1, 1, 1),
            (3, 3, 3)
        ),
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=None
    ):
        super(Edge_Module, self).__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.mid_channel = mid_channels
        self.kernel_sizes = kernel_sizes
        self.act = act
        self.norm = norm
        self.dropout = dropout

        self.conv1 = Convolution(
                dimensions=dimensions,
                in_channels=in_channels[0],
                out_channels=mid_channels,
                strides=1,
                kernel_size=kernel_sizes[0],
                act=self.act,
                norm=self.norm,
                dropout=self.dropout
            )

        self.conv2 = Convolution(
                dimensions=dimensions,
                in_channels=in_channels[1],
                out_channels=mid_channels,
                strides=1,
                kernel_size=kernel_sizes[0],
                act=self.act,
                norm=self.norm,
                dropout=self.dropout
            )

        self.conv3 = Convolution(
                dimensions=dimensions,
                in_channels=in_channels[2],
                out_channels=mid_channels,
                strides=1,
                kernel_size=kernel_sizes[0],
                act=self.act,
                norm=self.norm,
                dropout=self.dropout
            )

        self.conv3_1 = Convolution(
                dimensions=dimensions,
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=1,
                kernel_size=kernel_sizes[1],
                act=self.act,
                norm=self.norm,
                dropout=self.dropout
            )

        self.conv3_2 = Convolution(
                dimensions=dimensions,
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=1,
                kernel_size=kernel_sizes[1],
                act=self.act,
                norm=self.norm,
                dropout=self.dropout
            )

        self.conv3_3 = Convolution(
                dimensions=dimensions,
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=1,
                kernel_size=kernel_sizes[1],
                act=self.act,
                norm=self.norm,
                dropout=self.dropout
            )

        self.classifer = Convolution(
                dimensions=dimensions,
                in_channels=mid_channels*3,
                out_channels=1,
                strides=1,
                kernel_size=kernel_sizes[1],
                act=None,
                norm=None,
                dropout=self.dropout
            )

        self.rcab = RCAB(mid_channels * 3)

    def forward(self, fea_maps):
        # import pdb;pdb.set_trace()


        _, _, x3, x2, _, x1 = fea_maps

        _, _, h, w, d = x1.size() # 224, 224, 48
        edge1_fea = self.conv1(x1)
        edge1 = self.conv3_1(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv3_2(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv3_3(edge3_fea)
        

        edge2 = F.interpolate(edge2, size=(h, w, d), mode='trilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w, d), mode='trilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1) # B*48*224*224*48
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        # import pdb;pdb.set_trace()
        return edge

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, 
        channels, 
        dimensions=3,
        kernel_size=(3, 3, 3), 
        reduction=4,
        bias=True, 
        act=Act.RELU
    ):

        super(RCAB, self).__init__()
        modules_body = [] # conv, relu, conv, CAlayer
        modules_body.append(
            Convolution(
                dimensions=dimensions,
                in_channels=channels,
                out_channels=channels,
                strides=1,
                kernel_size=kernel_size,
                act=Act.RELU,
                norm=None,
                dropout=None,
                bias= bias
            )
        )
        modules_body.append(
            Convolution(
                dimensions=dimensions,
                in_channels=channels,
                out_channels=channels,
                strides=1,
                kernel_size=kernel_size,
                act=None,
                norm=None,
                dropout=None,
                bias= bias
            )
        )
        modules_body.append(CALayer(channels, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x): # B*24*224*224*48
        res = self.body(x)
        # import pdb;pdb.set_trace()
        res += x
        return res

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(
        self, 
        channels, 
        reduction=4,
        dimensions=3,
        kernel_size=(1,1,1),
        bias=True
    ):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            Convolution(
                dimensions=dimensions,
                in_channels=channels,
                out_channels=channels,
                strides=1,
                kernel_size=kernel_size,
                act=Act.RELU,
                norm=None,
                dropout=None,
                bias= bias
            ),
            Convolution(
                dimensions=dimensions,
                in_channels=channels,
                out_channels=channels,
                strides=1,
                kernel_size=kernel_size,
                act=Act.SIGMOID,
                norm=None,
                dropout=None,
                bias= bias
            )
        )

    def forward(self, x):
        y = self.avg_pool(x)
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        y = self.conv_du(y)
        return x * y


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            Convolution(
                dimensions=3,
                in_channels=in_dim,
                out_channels=reduction_dim,
                strides=1,
                kernel_size=(1,1,1),
                act=Act.RELU,
                norm=None,
                bias=False
            )
        )
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv3d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool3d(1)
        self.img_conv = Convolution(
                dimensions=3,
                in_channels=in_dim,
                out_channels=reduction_dim,
                strides=1,
                kernel_size=(1,1,1),
                act=Act.RELU,
                norm=None,
                bias=False
        )
        self.edge_conv = Convolution(
                dimensions=3,
                in_channels=1,
                out_channels=reduction_dim,
                strides=1,
                kernel_size=(1,1,1),
                act=Act.RELU,
                norm=None,
                bias=False
        )


    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        # import pdb;pdb.set_trace()
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='trilinear', align_corners=True)  

        edge_features = self.edge_conv(edge)
        edge_features = F.interpolate(edge_features, x_size[2:],
                                      mode='trilinear', align_corners=True)

        out = torch.cat((img_features, edge_features), 1)
        # import pdb;pdb.set_trace()

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out
