# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union
import math 
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep


__all__ = ["BasicUnet", "Basicunet", "basicunet", "BasicUNet"]

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        self.temb_proj = torch.nn.Linear(512,
                                         out_chns)
        self.dim = spatial_dims
        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)
    
    def forward(self, x, temb):
        x = self.conv_0(x)
        if self.dim == 3:
            x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        elif self.dim == 2:
            x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        x = self.conv_1(x)
        return x 

class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

    def forward(self, x, temb):
        x = self.max_pooling(x)
        x = self.convs(x, temb)
        return x 

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), temb)  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0, temb)

        return x


class BasicUNetDe(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t, embeddings=None, image=None):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None :
            x = torch.cat([image, x], dim=1)
            
        x0 = self.conv_0(x, temb)
        if embeddings is not None:
            x0 += embeddings[0]

        x1 = self.down_1(x0, temb)
        if embeddings is not None:
            x1 += embeddings[1]

        x2 = self.down_2(x1, temb)
        if embeddings is not None:
            x2 += embeddings[2]

        x3 = self.down_3(x2, temb)
        if embeddings is not None:
            x3 += embeddings[3]

        x4 = self.down_4(x3, temb)
        if embeddings is not None:
            x4 += embeddings[4]

        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits


class GatedFusion(nn.Module):
    def __init__(self, channels, reduction=4, gate_init=1.0):
        super().__init__()

        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels * 3, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_gate(gate_init)

    def _init_gate(self, gate_init):
        """
        gate_init = 1.0 -> gate ≈ 1
        gate_init = 0.5 -> neutral
        gate_init = 0.0 -> close to 0
        """
        # Sigmoid inverse
        eps = 1e-6
        gate_init = min(max(gate_init, eps), 1 - eps)
        bias_value = torch.log(torch.tensor(gate_init / (1 - gate_init)))

        last_conv = self.controller[-2]  # Conv3d before Sigmoid

        last_conv = self.controller[-2]
        nn.init.zeros_(last_conv.weight)
        nn.init.constant_(last_conv.bias, 10.0)  # sigmoid(10) ≈ 0.99995

    def forward(self, x, emb1, emb2, need_weight = False):

        # 2. 收集所有信息源
        combined = torch.cat([x, emb1, emb2], dim=1)

        # 3. 运行“控制器”得到门控信号
        gates = self.controller(combined)

        # 4. 分离出各自的门控
        gate1, gate2 = gates.chunk(2, dim=1)

        # 5. 应用门控并融合

        fused_feature = x + gate1 * emb1 + gate2 * emb2
        if need_weight:
            return fused_feature, gate1, gate2
        else:
            return fused_feature

class BasicUNetDe_GateFusion(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.gate_fuser_0 = GatedFusion(features[0])
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.gate_fuser_1 = GatedFusion(features[1])
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.gate_fuser_2 = GatedFusion(features[2])
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.gate_fuser_3 = GatedFusion(features[3])
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.gate_fuser_4 = GatedFusion(features[4])

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, t, embeddings=None, embeddings2 = None, image=None,
                writer = None, iter_num = None # visualization for gate weight
                ):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, temb)
        if writer==None:
            if embeddings is not None:
                x0 = self.gate_fuser_0(x0, embeddings[0], embeddings2[0])
            x1 = self.down_1(x0, temb)
            if embeddings is not None:
                x1 = self.gate_fuser_1(x1, embeddings[1], embeddings2[1])
            x2 = self.down_2(x1, temb)
            if embeddings is not None:
                x2 = self.gate_fuser_2(x2, embeddings[2], embeddings2[2])
            x3 = self.down_3(x2, temb)
            if embeddings is not None:
                x3 = self.gate_fuser_3(x3, embeddings[3], embeddings2[3])
            x4 = self.down_4(x3, temb)
            if embeddings is not None:
                x4 = self.gate_fuser_4(x4, embeddings[4], embeddings2[4])
        else:
            x0, g0_1, g0_2 = self.gate_fuser_0(x0, embeddings[0], embeddings2[0],need_weight = True)
            x1 = self.down_1(x0, temb)
            x1, g1_1, g1_2 = self.gate_fuser_1(x1, embeddings[1], embeddings2[1],need_weight = True)
            x2 = self.down_2(x1, temb)
            x2, g2_1, g2_2 = self.gate_fuser_2(x2, embeddings[2], embeddings2[2],need_weight = True)
            x3 = self.down_3(x2, temb)
            x3, g3_1, g3_2 = self.gate_fuser_3(x3, embeddings[3], embeddings2[3],need_weight = True)
            x4 = self.down_4(x3, temb)
            x4, g4_1, g4_2 = self.gate_fuser_4(x4, embeddings[4], embeddings2[4],need_weight = True)
            gates_1 = [g0_1, g1_1, g2_1, g3_1, g4_1]
            gates_2 = [g0_2, g1_2, g2_2, g3_2, g4_2]
            for i, (g1, g2) in enumerate(zip(gates_1, gates_2)):
                mean1, std1 = self.compute_gate_stats(g1)
                mean2, std2 = self.compute_gate_stats(g2)

                # ===== TensorBoard 可视化（示例）=====
                if writer is not None and iter_num is not None:
                    writer.add_scalar(f"gate/layer{i}_emb1_mean", mean1.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb1_std", std1.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb2_mean", mean2.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb2_std", std2.mean().item(), iter_num)


        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits

    def compute_gate_stats(self, gate: torch.Tensor):
        """
        Compute Mean ± Std for gate tensor
        """

        gate = gate.squeeze()

        mean = gate.mean(dim=0)   # [C]
        std = gate.std(dim=0)     # [C]
        return mean, std


# Ablation Studies

class GatedFusion_removeFteacher(nn.Module):
    def __init__(self, channels, reduction=4, gate_init=1.0):
        super().__init__()

        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels * 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_gate(gate_init)

    def _init_gate(self, gate_init):
        """
        gate_init = 1.0 -> gate ≈ 1
        gate_init = 0.5 -> neutral
        gate_init = 0.0 -> close to 0
        """
        # Sigmoid inverse
        eps = 1e-6
        gate_init = min(max(gate_init, eps), 1 - eps)
        bias_value = torch.log(torch.tensor(gate_init / (1 - gate_init)))

        last_conv = self.controller[-2]  # Conv3d before Sigmoid

        last_conv = self.controller[-2]
        nn.init.zeros_(last_conv.weight)
        nn.init.constant_(last_conv.bias, 10.0)  # sigmoid(10) ≈ 0.99995

    def forward(self, x, emb1, emb2, need_weight = False):

        # 2. 收集所有信息源
        combined = torch.cat([x, emb1], dim=1)

        # 3. 运行“控制器”得到门控信号
        gate = self.controller(combined)


        fused_feature = x + gate * emb1
        if need_weight:
            return fused_feature, gate, torch.zeros_like(gate)
        else:
            return fused_feature

class BasicUNetDe_GateFusion_removeFteacher(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.gate_fuser_0 = GatedFusion_removeFteacher(features[0])
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.gate_fuser_1 = GatedFusion_removeFteacher(features[1])
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.gate_fuser_2 = GatedFusion_removeFteacher(features[2])
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.gate_fuser_3 = GatedFusion_removeFteacher(features[3])
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.gate_fuser_4 = GatedFusion_removeFteacher(features[4])

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, t, embeddings=None, embeddings2 = None, image=None,
                writer = None, iter_num = None # visualization for gate weight
                ):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, temb)
        if writer==None:
            if embeddings is not None:
                x0 = self.gate_fuser_0(x0, embeddings[0], embeddings2[0])
            x1 = self.down_1(x0, temb)
            if embeddings is not None:
                x1 = self.gate_fuser_1(x1, embeddings[1], embeddings2[1])
            x2 = self.down_2(x1, temb)
            if embeddings is not None:
                x2 = self.gate_fuser_2(x2, embeddings[2], embeddings2[2])
            x3 = self.down_3(x2, temb)
            if embeddings is not None:
                x3 = self.gate_fuser_3(x3, embeddings[3], embeddings2[3])
            x4 = self.down_4(x3, temb)
            if embeddings is not None:
                x4 = self.gate_fuser_4(x4, embeddings[4], embeddings2[4])
        else:
            x0, g0_1, g0_2 = self.gate_fuser_0(x0, embeddings[0], embeddings2[0],need_weight = True)
            x1 = self.down_1(x0, temb)
            x1, g1_1, g1_2 = self.gate_fuser_1(x1, embeddings[1], embeddings2[1],need_weight = True)
            x2 = self.down_2(x1, temb)
            x2, g2_1, g2_2 = self.gate_fuser_2(x2, embeddings[2], embeddings2[2],need_weight = True)
            x3 = self.down_3(x2, temb)
            x3, g3_1, g3_2 = self.gate_fuser_3(x3, embeddings[3], embeddings2[3],need_weight = True)
            x4 = self.down_4(x3, temb)
            x4, g4_1, g4_2 = self.gate_fuser_4(x4, embeddings[4], embeddings2[4],need_weight = True)
            gates_1 = [g0_1, g1_1, g2_1, g3_1, g4_1]
            gates_2 = [g0_2, g1_2, g2_2, g3_2, g4_2]
            for i, (g1, g2) in enumerate(zip(gates_1, gates_2)):
                mean1, std1 = self.compute_gate_stats(g1)
                mean2, std2 = self.compute_gate_stats(g2)

                # ===== TensorBoard 可视化（示例）=====
                if writer is not None and iter_num is not None:
                    writer.add_scalar(f"gate/layer{i}_emb1_mean", mean1.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb1_std", std1.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb2_mean", mean2.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb2_std", std2.mean().item(), iter_num)


        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits

    def compute_gate_stats(self, gate: torch.Tensor):
        """
        Compute Mean ± Std for gate tensor
        """

        gate = gate.squeeze()

        mean = gate.mean(dim=0)   # [C]
        std = gate.std(dim=0)     # [C]
        return mean, std

class GatedFusion_removeFimage(nn.Module):
    def __init__(self, channels, reduction=4, gate_init=1.0):
        super().__init__()

        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels * 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_gate(gate_init)

    def _init_gate(self, gate_init):
        """
        gate_init = 1.0 -> gate ≈ 1
        gate_init = 0.5 -> neutral
        gate_init = 0.0 -> close to 0
        """
        # Sigmoid inverse
        eps = 1e-6
        gate_init = min(max(gate_init, eps), 1 - eps)
        bias_value = torch.log(torch.tensor(gate_init / (1 - gate_init)))

        last_conv = self.controller[-2]  # Conv3d before Sigmoid

        last_conv = self.controller[-2]
        nn.init.zeros_(last_conv.weight)
        nn.init.constant_(last_conv.bias, 10.0)  # sigmoid(10) ≈ 0.99995

    def forward(self, x, emb1, emb2, need_weight = False):

        # 2. 收集所有信息源
        combined = torch.cat([x, emb2], dim=1)

        # 3. 运行“控制器”得到门控信号
        gate = self.controller(combined)

        # 5. 应用门控并融合

        fused_feature = x + gate * emb2
        if need_weight:
            return fused_feature, torch.zeros_like(gate), gate
        else:
            return fused_feature

class BasicUNetDe_GateFusion_removeFimage(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.gate_fuser_0 = GatedFusion_removeFimage(features[0])
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.gate_fuser_1 = GatedFusion_removeFimage(features[1])
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.gate_fuser_2 = GatedFusion_removeFimage(features[2])
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.gate_fuser_3 = GatedFusion_removeFimage(features[3])
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.gate_fuser_4 = GatedFusion_removeFimage(features[4])

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, t, embeddings=None, embeddings2 = None, image=None,
                writer = None, iter_num = None # visualization for gate weight
                ):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, temb)
        if writer==None:
            if embeddings is not None:
                x0 = self.gate_fuser_0(x0, embeddings[0], embeddings2[0])
            x1 = self.down_1(x0, temb)
            if embeddings is not None:
                x1 = self.gate_fuser_1(x1, embeddings[1], embeddings2[1])
            x2 = self.down_2(x1, temb)
            if embeddings is not None:
                x2 = self.gate_fuser_2(x2, embeddings[2], embeddings2[2])
            x3 = self.down_3(x2, temb)
            if embeddings is not None:
                x3 = self.gate_fuser_3(x3, embeddings[3], embeddings2[3])
            x4 = self.down_4(x3, temb)
            if embeddings is not None:
                x4 = self.gate_fuser_4(x4, embeddings[4], embeddings2[4])
        else:
            x0, g0_1, g0_2 = self.gate_fuser_0(x0, embeddings[0], embeddings2[0],need_weight = True)
            x1 = self.down_1(x0, temb)
            x1, g1_1, g1_2 = self.gate_fuser_1(x1, embeddings[1], embeddings2[1],need_weight = True)
            x2 = self.down_2(x1, temb)
            x2, g2_1, g2_2 = self.gate_fuser_2(x2, embeddings[2], embeddings2[2],need_weight = True)
            x3 = self.down_3(x2, temb)
            x3, g3_1, g3_2 = self.gate_fuser_3(x3, embeddings[3], embeddings2[3],need_weight = True)
            x4 = self.down_4(x3, temb)
            x4, g4_1, g4_2 = self.gate_fuser_4(x4, embeddings[4], embeddings2[4],need_weight = True)
            gates_1 = [g0_1, g1_1, g2_1, g3_1, g4_1]
            gates_2 = [g0_2, g1_2, g2_2, g3_2, g4_2]
            for i, (g1, g2) in enumerate(zip(gates_1, gates_2)):
                mean1, std1 = self.compute_gate_stats(g1)
                mean2, std2 = self.compute_gate_stats(g2)

                # ===== TensorBoard 可视化（示例）=====
                if writer is not None and iter_num is not None:
                    writer.add_scalar(f"gate/layer{i}_emb1_mean", mean1.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb1_std", std1.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb2_mean", mean2.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb2_std", std2.mean().item(), iter_num)


        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits

    def compute_gate_stats(self, gate: torch.Tensor):
        """
        Compute Mean ± Std for gate tensor
        """

        gate = gate.squeeze()

        mean = gate.mean(dim=0)   # [C]
        std = gate.std(dim=0)     # [C]
        return mean, std

class GatedFusion_removeFcombine(nn.Module):
    def __init__(self, channels, reduction=4, gate_init=1.0):
        super().__init__()

        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_gate(gate_init)

    def _init_gate(self, gate_init):
        """
        gate_init = 1.0 -> gate ≈ 1
        gate_init = 0.5 -> neutral
        gate_init = 0.0 -> close to 0
        """
        # Sigmoid inverse
        eps = 1e-6
        gate_init = min(max(gate_init, eps), 1 - eps)
        bias_value = torch.log(torch.tensor(gate_init / (1 - gate_init)))

        last_conv = self.controller[-2]  # Conv3d before Sigmoid

        last_conv = self.controller[-2]
        nn.init.zeros_(last_conv.weight)
        nn.init.constant_(last_conv.bias, 10.0)  # sigmoid(10) ≈ 0.99995

    def forward(self, x, emb1, emb2, need_weight = False):

        # 2. 收集所有信息源
        combined = torch.cat([emb1, emb2], dim=1)

        # 3. 运行“控制器”得到门控信号
        gates = self.controller(combined)

        # 4. 分离出各自的门控
        gate1, gate2 = gates.chunk(2, dim=1)

        # 5. 应用门控并融合

        fused_feature = gate1 * emb1 + gate2 * emb2
        if need_weight:
            return fused_feature, gate1, gate2
        else:
            return fused_feature


class BasicUNetDe_GateFusion_removeFcombine(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.gate_fuser_0 = GatedFusion_removeFcombine(features[0])
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.gate_fuser_1 = GatedFusion_removeFcombine(features[1])
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.gate_fuser_2 = GatedFusion_removeFcombine(features[2])
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.gate_fuser_3 = GatedFusion_removeFcombine(features[3])
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.gate_fuser_4 = GatedFusion_removeFcombine(features[4])

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, t, embeddings=None, embeddings2 = None, image=None,
                writer = None, iter_num = None # visualization for gate weight
                ):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, temb)
        if writer==None:
            if embeddings is not None:
                x0 = self.gate_fuser_0(x0, embeddings[0], embeddings2[0])
            x1 = self.down_1(x0, temb)
            if embeddings is not None:
                x1 = self.gate_fuser_1(x1, embeddings[1], embeddings2[1])
            x2 = self.down_2(x1, temb)
            if embeddings is not None:
                x2 = self.gate_fuser_2(x2, embeddings[2], embeddings2[2])
            x3 = self.down_3(x2, temb)
            if embeddings is not None:
                x3 = self.gate_fuser_3(x3, embeddings[3], embeddings2[3])
            x4 = self.down_4(x3, temb)
            if embeddings is not None:
                x4 = self.gate_fuser_4(x4, embeddings[4], embeddings2[4])
        else:
            x0, g0_1, g0_2 = self.gate_fuser_0(x0, embeddings[0], embeddings2[0],need_weight = True)
            x1 = self.down_1(x0, temb)
            x1, g1_1, g1_2 = self.gate_fuser_1(x1, embeddings[1], embeddings2[1],need_weight = True)
            x2 = self.down_2(x1, temb)
            x2, g2_1, g2_2 = self.gate_fuser_2(x2, embeddings[2], embeddings2[2],need_weight = True)
            x3 = self.down_3(x2, temb)
            x3, g3_1, g3_2 = self.gate_fuser_3(x3, embeddings[3], embeddings2[3],need_weight = True)
            x4 = self.down_4(x3, temb)
            x4, g4_1, g4_2 = self.gate_fuser_4(x4, embeddings[4], embeddings2[4],need_weight = True)
            gates_1 = [g0_1, g1_1, g2_1, g3_1, g4_1]
            gates_2 = [g0_2, g1_2, g2_2, g3_2, g4_2]
            for i, (g1, g2) in enumerate(zip(gates_1, gates_2)):
                mean1, std1 = self.compute_gate_stats(g1)
                mean2, std2 = self.compute_gate_stats(g2)

                # ===== TensorBoard 可视化（示例）=====
                if writer is not None and iter_num is not None:
                    writer.add_scalar(f"gate/layer{i}_emb1_mean", mean1.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb1_std", std1.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb2_mean", mean2.mean().item(), iter_num)
                    writer.add_scalar(f"gate/layer{i}_emb2_std", std2.mean().item(), iter_num)


        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits

    def compute_gate_stats(self, gate: torch.Tensor):
        """
        Compute Mean ± Std for gate tensor
        """

        gate = gate.squeeze()

        mean = gate.mean(dim=0)   # [C]
        std = gate.std(dim=0)     # [C]
        return mean, std


class BasicUNetDe_GateFusion_removeFcombine_IG(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.gate_fuser_0 = GatedFusion_removeFcombine(features[0])
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.gate_fuser_1 = GatedFusion_removeFcombine(features[1])
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.gate_fuser_2 = GatedFusion_removeFcombine(features[2])
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.gate_fuser_3 = GatedFusion_removeFcombine(features[3])
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.gate_fuser_4 = GatedFusion_removeFcombine(features[4])

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


    def forward(self, t, embeddings=None, embeddings2 = None,
                writer = None, iter_num = None # visualization for gate weight
                ):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)


        x0 = self.gate_fuser_0(None, embeddings[0], embeddings2[0])
        x1 = self.gate_fuser_1(None, embeddings[1], embeddings2[1])
        x2 = self.gate_fuser_2(None, embeddings[2], embeddings2[2])
        x3 = self.gate_fuser_3(None, embeddings[3], embeddings2[3])
        x4 = self.gate_fuser_4(None, embeddings[4], embeddings2[4])



        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits

    def compute_gate_stats(self, gate: torch.Tensor):
        """
        Compute Mean ± Std for gate tensor
        """

        gate = gate.squeeze()

        mean = gate.mean(dim=0)   # [C]
        std = gate.std(dim=0)     # [C]
        return mean, std

class BasicUNetDe_GateFusion_keepFcombine_only(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, t, embeddings=None, embeddings2 = None, image=None,
                writer = None, iter_num = None # visualization for gate weight
                ):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, temb)

        x1 = self.down_1(x0, temb)
        x2 = self.down_2(x1, temb)
        x3 = self.down_3(x2, temb)
        x4 = self.down_4(x3, temb)



        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits

    def compute_gate_stats(self, gate: torch.Tensor):
        """
        Compute Mean ± Std for gate tensor
        """

        gate = gate.squeeze()

        mean = gate.mean(dim=0)   # [C]
        std = gate.std(dim=0)     # [C]
        return mean, std




class FiLMFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # FiLM生成器（控制器）
        # 输入通道是 C*2 (来自emb1和emb2)，输出通道也是 C*2 (用于gamma和beta)
        self.generator = nn.Sequential(
            # 使用3x3卷积可以融合局部上下文信息，这对于图像特征很重要
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输出最终的gamma和beta
            nn.Conv3d(channels, channels * 2, kernel_size=3, padding=1)
        )

    def forward(self, x, emb1, emb2):
        # 1. 拼接两个外部特征作为条件信息
        # emb1 来自 V-Net, emb2 来自 U-Net
        context = torch.cat([emb1, emb2], dim=1)  # 形状: [B, C*2, H, W]

        # 2. 通过生成器，得到空间可变的 gamma 和 beta
        params = self.generator(context)  # 形状: [B, C*2, H, W]
        gamma, beta = params.chunk(2, dim=1)  # 每个形状: [B, C, H, W]

        # 3. 应用仿射变换进行调制
        fused_feature = gamma * x + beta

        return fused_feature


class BasicUNetDe_FiLMFusion(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.film_fuser_0 = FiLMFusion(features[0])
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.film_fuser_1 = FiLMFusion(features[1])
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.film_fuser_2 = FiLMFusion(features[2])
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.film_fuser_3 = FiLMFusion(features[3])
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.film_fuser_4 = FiLMFusion(features[4])

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor, t, embeddings=None, embeddings2 = None, image=None):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, temb)
        if embeddings is not None:
            x0 = self.film_fuser_0(x0,embeddings[0],embeddings2[0])

        x1 = self.down_1(x0, temb)
        if embeddings is not None:
            x1 = self.film_fuser_1(x1,embeddings[1],embeddings2[1])

        x2 = self.down_2(x1, temb)
        if embeddings is not None:
            x2 = self.film_fuser_2(x2,embeddings[2],embeddings2[2])

        x3 = self.down_3(x2, temb)
        if embeddings is not None:
            x3 = self.film_fuser_3(x3, embeddings[3], embeddings2[3])

        x4 = self.down_4(x3, temb)
        if embeddings is not None:
            x4 = self.film_fuser_4(x4, embeddings[4], embeddings2[4])

        u4 = self.upcat_4(x4, x3, temb)
        u3 = self.upcat_3(u4, x2, temb)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits

class CrossAttention(nn.Module):
    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: int = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor = None):
        q = self.to_q(hidden_states)
        context_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.to_k(context_states)
        v = self.to_v(context_states)

        b, _, _ = q.shape
        q = q.view(b, -1, self.heads, self.inner_dim // self.heads).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.inner_dim // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.inner_dim // self.heads).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        hidden_states = torch.matmul(attention_probs, v)

        hidden_states = hidden_states.transpose(1, 2).reshape(b, -1, self.inner_dim)
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class UpCatCrossAttention(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            context_dim: int,  # 明确接收 context_dim
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple],
            upsample: str,
            halves: bool = True,
            heads: int = 8,
            dim_head: int = 64
    ):
        super().__init__()
        self.upsample = UpSample(spatial_dims, in_channels, out_channels, scale_factor=2, mode=upsample, pre_conv=None)
        self.convs = TwoConv(
            spatial_dims,
            (out_channels if halves else in_channels) + skip_channels,
            out_channels,
            act,
            norm,
            bias,
            dropout,
        )

        if dim_head == -1:
            if out_channels % heads != 0:
                raise ValueError(f"out_channels ({out_channels}) must be divisible by heads ({heads})")
            dim_head = out_channels // heads

        self.attn = CrossAttention(
            query_dim=out_channels,
            cross_attention_dim=context_dim,  # 使用传入的 context_dim
            heads=heads,
            dim_head=dim_head,
            dropout=float(dropout) if isinstance(dropout, (int, float)) else 0.0
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor, temb: torch.Tensor, context: torch.Tensor):
        x_up = self.upsample(x)
        x = torch.cat([x_skip, x_up], dim=1)
        x = self.convs(x, temb)

        b, c, *spatial_dims_list = x.shape
        x_seq = x.view(b, c, -1).permute(0, 2, 1)

        attn_output = self.attn(
            hidden_states=x_seq,
            encoder_hidden_states=context
        )

        attn_output = attn_output.permute(0, 2, 1).view(b, c, *spatial_dims_list)
        return x + attn_output


class BasicUNetDe_crossAttention(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (16, 32, 64, 128, 256, 32),
            context_dim: int = 256,  # 方案A的关键：一个固定的全局上下文维度
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # --- 时间步嵌入 (不变) ---
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(128, 512),
            nn.Linear(512, 512),
        ])

        # --- 编码器 (不变) ---
        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        # --- 解码器 (创建时传入固定的 context_dim) ---
        self.upcat_4 = UpCatCrossAttention(spatial_dims, fea[4], fea[3], fea[3], context_dim, act, norm, bias, dropout,
                                           upsample)
        self.upcat_3 = UpCatCrossAttention(spatial_dims, fea[3], fea[2], fea[2], context_dim, act, norm, bias, dropout,
                                           upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        # --- 输出层 (不变) ---
        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t, embeddings=None, image=None):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            x = torch.cat([image, x], dim=1)

        x0 = self.conv_0(x, temb)
        x1 = self.down_1(x0, temb)
        x2 = self.down_2(x1, temb)
        x3 = self.down_3(x2, temb)
        x4 = self.down_4(x3, temb)

        if embeddings is None:
            raise ValueError("Cross-attention UNet requires `embeddings` (context) to be provided.")

        # --- 方案A的核心逻辑 ---
        # 1. 假设 embeddings 是最深层的特征图或已经处理好的序列
        #    为了健壮性，我们检查它的维度
        if isinstance(embeddings, list):
            # 如果传入的是列表，我们只取最深层的
            context_map = embeddings[-1]
        else:
            # 否则，我们假设它就是我们需要的上下文
            context_map = embeddings

        # 2. 确保上下文是序列格式 (B, N, D)
        if context_map.dim() > 3:
            # 如果是特征图 (B, C, H, W, ...)，则转换为序列
            b, c, *spatial = context_map.shape
            context_sequence = context_map.view(b, c, -1).permute(0, 2, 1)
        elif context_map.dim() == 3:
            # 如果已经是序列，直接使用
            context_sequence = context_map
        else:
            raise ValueError(f"Unsupported embeddings shape: {context_map.shape}")

        # 3. 将这个单一的序列传递给所有层
        u4 = self.upcat_4(x4, x3, temb, context_sequence)
        u3 = self.upcat_3(u4, x2, temb, context_sequence)
        u2 = self.upcat_2(u3, x1, temb)
        u1 = self.upcat_1(u2, x0, temb)

        logits = self.final_conv(u1)
        return logits




