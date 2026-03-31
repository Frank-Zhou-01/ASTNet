import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from mmcv.cnn import ConvModule


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    # for m in module.modules():
    #     if isinstance(m, nn.Conv2d):
    #         kaiming_init(m, a=0, mode='fan_in', bias=0)
    #         m.weight.data *= scale
        # elif isinstance(m, nn.Linear):
        #     kaiming_init(m, a=0, mode='fan_in', bias=0)
        #     m.weight.data *= scale
        # elif isinstance(m, _BatchNorm):
        #     constant_init(m.weight, val=1, bias=0)
    pass


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Note that in this function, the images are already resized to a
        multiple of 32.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.
        This function computes the optical flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].
        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        return self.fn(x, *args, **kwargs)


# ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        # residual blocks
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.act = act_layer()

    def forward(self, x):
        input = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return input + self.conv2(self.act(self.conv1(x)))


# Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, dim, num_resblocks):
        super().__init__()
        main = []
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_resblocks, mid_channels=dim))
        self.net = nn.Sequential(*main)

    def forward(self, x):
        out = self.net(x)
        return out


# Flow-Guided Sparse Window-based Multi-head Self-Attention
class FGSW_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(5, 4, 4),
            dim_head=64,
            heads=8,
            shift=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift
        inner_dim = dim_head * heads

        # position embedding
        q_l = self.window_size[1] * self.window_size[2]
        kv_l = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.static_a = nn.Parameter(torch.Tensor(1, heads, q_l, kv_l))
        trunc_normal_(self.static_a)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 3, 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 3, 1, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)

    def forward(self, q_inp, k_inp, flow):
        """
        :param q_inp: [n,1,c,h,w]
        :param k_inp: [n,2r+1,c,h,w]  (r: temporal radius of neighboring frames)
        :param flow: list: [[n,2,h,w],[n,2,h,w]]
        :return: out: [n,1,c,h,w]
        """
        b, f_q, c, h, w = q_inp.shape
        fb, hb, wb = self.window_size

        [flow_f, flow_b] = flow
        # sliding window
        if self.shift:
            q_inp, k_inp = map(lambda x: torch.roll(x, shifts=(-hb // 2, -wb // 2), dims=(-2, -1)), (q_inp, k_inp))
            if flow_f is not None:
                flow_f = torch.roll(flow_f, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
            if flow_b is not None:
                flow_b = torch.roll(flow_b, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
        k_f, k_r, k_b = k_inp[:, 0], k_inp[:, 1], k_inp[:, 2]

        # retrive key elements
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False
        grid = grid.type_as(k_f)
        if flow_f is not None:
            vgrid = grid + flow_f.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            # index the nearest token
            # k_f = F.grid_sample(k_f.float(), vgrid_scaled, mode='bilinear')
            k_f = F.grid_sample(k_f.float(), vgrid_scaled, mode='nearest')
        if flow_b is not None:
            vgrid = grid + flow_b.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            # index the nearest token
            # k_b = F.grid_sample(k_b.float(), vgrid_scaled, mode='bilinear')
            k_b = F.grid_sample(k_b.float(), vgrid_scaled, mode='nearest')

        k_inp = torch.stack([k_f, k_r, k_b], dim=1)
        # norm
        q = self.norm_q(q_inp.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        kv = self.norm_kv(k_inp.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        q = self.to_q(q.flatten(0, 1))
        k, v = self.to_kv(kv.flatten(0, 1)).chunk(2, dim=1)

        # split into (B,N,C)
        q, k, v = map(lambda t: rearrange(t, '(b f) c (h p1) (w p2)-> (b h w) (f p1 p2) c', p1=hb, p2=wb, b=b),
                      (q, k, v))

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # scale
        q *= self.scale

        # attention
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.static_a
        attn = sim.softmax(dim=-1)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        # merge windows back to original feature map
        out = rearrange(out, '(b h w) (f p1 p2) c -> (b f) c (h p1) (w p2)', b=b, h=(h // hb), w=(w // wb),
                        p1=hb, p2=wb)

        # combine heads
        out = self.to_out(out).view(b, f_q, c, h, w)

        # inverse shift
        if self.shift:
            out = torch.roll(out, shifts=(hb // 2, wb // 2), dims=(-2, -1))

        return out


class FGAB(nn.Module):
    def __init__(
            self,
            q_dim,
            emb_dim,
            window_size=(3, 4, 4),
            dim_head=64,
            heads=8,
            num_resblocks=5,
            shift=False
    ):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.embed_dim = emb_dim
        self.q_dim = q_dim
        self.attn = FGSW_MSA(q_dim, window_size, dim_head, heads, shift=shift)
        self.feed_forward = FeedForward(q_dim, num_resblocks)
        self.conv = nn.Conv2d(q_dim + emb_dim, q_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.shift = shift

    def forward(self, x, flows_forward, flows_backward, cpu_cache):
        """
        :param x: [n,t,c,h,w]
        :param flows_forward: [n,t,2,h,w]
        :param flows_backward: [n,t,2,h,w]
        :return: outs: [n,t,c,h,w]
        """
        t = len(x)
        n, c, h, w = x[0].shape
        outs = []
        embedding = flows_forward[0].new_zeros(n, self.embed_dim, h, w)
        for i in range(0, t):
            flow_f, flow_b = None, None
            if i > 0:
                flow_f = flows_forward[i - 1]
                if cpu_cache:
                    flow_f = flow_f.cuda()
                    embedding = embedding.cuda()
                embedding = flow_warp(embedding, flow_f.permute(0, 2, 3, 1))
                k_f = x[i - 1]
            else:
                k_f = x[i]
            if i < t - 1:
                flow_b = flows_backward[i]
                if cpu_cache:
                    flow_b = flow_b.cuda()
                k_b = x[i + 1]
            else:
                k_b = x[i]
            x_current = x[i]
            if cpu_cache:
                embedding = embedding.cuda()
                x_current = x_current.cuda()
                k_f = k_f.cuda()
                k_b = k_b.cuda()
            q_inp = self.lrelu(self.conv(torch.cat((embedding, x_current), dim=1))).unsqueeze(1)
            k_inp = torch.stack([k_f, x_current, k_b], dim=1)
            out = self.attn(q_inp=q_inp, k_inp=k_inp, flow=[flow_f, flow_b]) + q_inp
            out = out.squeeze(1)
            out = self.feed_forward(out) + out
            embedding = out
            if cpu_cache:
                out = out.cpu()
                torch.cuda.empty_cache()
            outs.append(out)
        return outs


class FGABs(nn.Module):
    def __init__(
            self,
            q_dim,
            emb_dim,
            window_size=(3, 3, 3),
            heads=4,
            dim_head=32,
            num_resblocks=20,
            num_FGAB=1,
            reverse=(True, False),
            shift=(True, False)

    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FGAB(
                q_dim=q_dim, emb_dim=emb_dim, window_size=window_size, heads=heads, dim_head=dim_head,
                num_resblocks=num_resblocks, shift=shift
            )
            for _ in range(num_FGAB)])
        self.reverse = reverse

    def forward(self, video, flows_forward, flows_backward, cpu_cache):
        """
        :param video: [n,t,c,h,w]
        :param flows: list [[n,t-1,2,h,w],[n,t-1,2,h,w]]
        :return: x: [n,t,c,h,w]
        """
        for i in range(len(self.layers)):
            layer = self.layers[i]
            reverse = self.reverse[i]
            if not reverse:
                video = layer(video, flows_forward=flows_forward, flows_backward=flows_backward, cpu_cache=cpu_cache)
            else:
                video = layer(video[::-1], flows_forward=flows_backward[::-1],
                              flows_backward=flows_forward[::-1], cpu_cache=cpu_cache)
                video = video[::-1]
        return video


def Forward(x, model, cpu_cache):
    feat = []
    t = len(x)
    for i in range(0, t):
        feat_i = x[i]
        if cpu_cache:
            feat_i = feat_i.cuda()
        feat_i = model(feat_i)
        if cpu_cache:
            feat_i = feat_i.cpu()
            torch.cuda.empty_cache()
        feat.append(feat_i)
    return feat


class FGST(nn.Module):
    def __init__(self, dim=32, spynet_pretrained=None, cpu_cache_length=30, patch_test=False):
        super(FGST, self).__init__()
        self.dim = dim
        self.cpu_cache_length = cpu_cache_length
        self.patch_test = patch_test

        #### optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        #### embedding
        self.embedding = ResidualBlocksWithInputConv(3, dim, 5)

        #### transformer blocks
        self.window_size = [3, 3, 3]
        self.emb_dim = 32
        self.encoder_1 = FGABs(
            q_dim=self.dim,
            emb_dim=self.emb_dim,
            window_size=self.window_size,
            num_FGAB=1,
            heads=2,
            dim_head=32,
            num_resblocks=5,
            reverse=[False],
            shift=[False]
        )
        self.parchmerge_1 = nn.Conv2d(self.dim, self.dim * 2, 4, 2, 1, bias=False)
        self.encoder_2 = FGABs(
            q_dim=self.dim * 2,
            emb_dim=self.emb_dim * 2,
            window_size=self.window_size,
            num_FGAB=1,
            heads=4,
            dim_head=32,
            num_resblocks=5,
            reverse=[False],
            shift=[False]
        )
        self.parchmerge_2 = nn.Conv2d(self.dim * 2, self.dim * 4, 4, 2, 1, bias=False)
        self.bottle_neck = FGABs(
            q_dim=self.dim * 4,
            emb_dim=self.emb_dim * 4,
            window_size=self.window_size,
            num_FGAB=2,
            heads=8,
            dim_head=32,
            num_resblocks=5,
            reverse=[False, True],
            shift=[False, True]
        )
        self.patchexpand_1 = nn.ConvTranspose2d(self.dim * 4, self.dim * 2, stride=2, kernel_size=2, padding=0,
                                                output_padding=0)
        self.fution_1 = nn.Conv2d(self.dim * 4, self.dim * 2, 3, 1, 1, bias=False)
        self.decoder_1 = FGABs(
            q_dim=self.dim * 2,
            emb_dim=self.emb_dim * 2,
            window_size=self.window_size,
            num_FGAB=1,
            heads=4,
            dim_head=32,
            num_resblocks=5,
            reverse=[True],
            shift=[True]
        )
        self.patchexpand_2 = nn.ConvTranspose2d(self.dim * 2, self.dim * 1, stride=2, kernel_size=2, padding=0,
                                                output_padding=0)
        self.fution_2 = nn.Conv2d(self.dim * 2, self.dim, 3, 1, 1, bias=False)
        self.decoder_2 = FGABs(
            q_dim=self.dim,
            emb_dim=self.emb_dim,
            window_size=self.window_size,
            num_FGAB=1,
            heads=2,
            dim_head=32,
            num_resblocks=5,
            reverse=[True],
            shift=[True]
        )
        # residual blocks after transformer
        main = []
        main.append(make_layer(ResidualBlockNoBN, 5, mid_channels=dim))
        main.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(nn.Conv2d(self.dim, 3, 3, 1, 1, bias=True))
        self.tail = nn.Sequential(*main)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def spatial_padding(self, lqs, pad_size=(12, 12)):

        n, t, c, h, w = lqs.shape
        (hb, wb) = pad_size
        pad_h = (hb - h % hb) % hb
        pad_w = (wb - w % wb) % wb

        # padding
        lqs = lqs.view(-1, c, h, w)
        lqs = F.pad(lqs, [0, pad_w, 0, pad_h], mode='reflect')

        return lqs.view(n, t, c, h + pad_h, w + pad_w)

    def compute_flow(self, lqs, flows):
        n, t, c, h, w = lqs.size()
        flows['forward'], flows['backward'], flows['forward_ds2'], flows['backward_ds2'], \
            flows['forward_ds4'], flows['backward_ds4'] = [], [], [], [], [], []
        lqs_1 = lqs[:, :-1, :, :, :]
        lqs_2 = lqs[:, 1:, :, :, :]
        for i in range(t - 1):
            lq_1, lq_2 = lqs_1[:, i, :, :, :], lqs_2[:, i, :, :, :]
            flow_backward_ = self.spynet(lq_1, lq_2)
            flow_forward_ = self.spynet(lq_2, lq_1)
            flow_backward_ds2_ = F.avg_pool2d(flow_backward_, kernel_size=2, stride=2) / 2.0
            flow_forward_ds2_ = F.avg_pool2d(flow_forward_, kernel_size=2, stride=2) / 2.0
            flow_backward_ds4_ = F.avg_pool2d(flow_backward_ds2_, kernel_size=2, stride=2) / 2.0
            flow_forward_ds4_ = F.avg_pool2d(flow_forward_ds2_, kernel_size=2, stride=2) / 2.0
            if self.cpu_cache:
                flow_backward_, flow_forward_ = flow_backward_.cpu(), flow_forward_.cpu()
                flow_backward_ds2_, flow_forward_ds2_ = flow_backward_ds2_.cpu(), flow_forward_ds2_.cpu()
                flow_backward_ds4_, flow_forward_ds4_ = flow_backward_ds4_.cpu(), flow_forward_ds4_.cpu()
                torch.cuda.empty_cache()
            flows['forward'].append(flow_forward_)
            flows['backward'].append(flow_backward_)
            flows['forward_ds2'].append(flow_forward_ds2_)
            flows['backward_ds2'].append(flow_backward_ds2_)
            flows['forward_ds4'].append(flow_forward_ds4_)
            flows['backward_ds4'].append(flow_backward_ds4_)
        return flows

    def forward(self, x):
        """
        :param x: [n,t,c,h,w]
        :return: out: [n,t,c,h,w]
        """
        n, t, c, h_input, w_input = x.size()
        if self.patch_test and (h_input, w_input) != (256, 256):
            # pad the input and make sure that it can be reshape into several windows
            x = self.spatial_padding(x, pad_size=(256, 256))
            h_pad, w_pad = x.shape[-2], x.shape[-1]
            x = rearrange(x, 'n t c (h p1) (w p2)-> (n h w) t c p1 p2', p1=256, p2=256)
        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and x.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        # pad the input and make sure that it can be reshape into several windows
        lqs = self.spatial_padding(x)
        h, w = lqs.size(3), lqs.size(4)

        # compute optical flow
        flows = {}
        flows = self.compute_flow(lqs, flows)

        feats = {}
        # embedding
        if self.cpu_cache:
            feats['encoder1'] = []
            for i in range(0, t):
                feat_ = self.embedding(lqs[:, i, :, :, :]).cpu()
                feats['encoder1'].append(feat_)
                torch.cuda.empty_cache()
        else:
            feats_ = self.embedding(lqs.flatten(0, 1)).view(n, t, self.dim, h, w)
            feats['encoder1'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # FGABs
        feats['encoder1'] = self.encoder_1(feats['encoder1'], flows['forward'], flows['backward'], self.cpu_cache)
        feats['encoder2'] = Forward(feats['encoder1'], self.parchmerge_1, self.cpu_cache)
        feats['encoder2'] = self.encoder_2(feats['encoder2'], flows['forward_ds2'], flows['backward_ds2'],
                                           self.cpu_cache)

        feats['bottle'] = Forward(feats['encoder2'], self.parchmerge_2, self.cpu_cache)
        feats['bottle'] = self.bottle_neck(feats['bottle'], flows['forward_ds4'], flows['backward_ds4'], self.cpu_cache)
        feats['decoder1'] = Forward(feats['bottle'], self.patchexpand_1, self.cpu_cache)
        if self.cpu_cache:
            del feats['bottle']

        for i in range(0, t):
            feat_encoder_2 = feats['encoder2'][i]
            feat_decoder_1 = feats['decoder1'][i]
            if self.cpu_cache:
                feat_encoder_2 = feat_encoder_2.cuda()
                feat_decoder_1 = feat_decoder_1.cuda()
            feat_ = self.lrelu(self.fution_1(torch.cat((feat_encoder_2, feat_decoder_1), dim=1)))
            if self.cpu_cache:
                feat_ = feat_.cpu()
                torch.cuda.empty_cache()
            feats['decoder1'][i] = feat_
        if self.cpu_cache:
            del feats['encoder2']

        feats['decoder1'] = self.decoder_1(feats['decoder1'], flows['forward_ds2'], flows['backward_ds2'],
                                           self.cpu_cache)
        feats['decoder2'] = Forward(feats['decoder1'], self.patchexpand_2, self.cpu_cache)
        if self.cpu_cache:
            del feats['decoder1'], flows['forward_ds2'], flows['backward_ds2']

        for i in range(0, t):
            feat_encoder_1 = feats['encoder1'][i]
            feat_decoder_2 = feats['decoder2'][i]
            if self.cpu_cache:
                feat_encoder_1 = feat_encoder_1.cuda()
                feat_decoder_2 = feat_decoder_2.cuda()
            feat_ = self.lrelu(self.fution_2(torch.cat((feat_encoder_1, feat_decoder_2), dim=1)))
            if self.cpu_cache:
                feat_ = feat_.cpu()
                torch.cuda.empty_cache()
            feats['decoder2'][i] = feat_
        if self.cpu_cache:
            del feats['encoder1']
        feats['decoder2'] = self.decoder_2(feats['decoder2'], flows['forward'], flows['backward'], self.cpu_cache)
        if self.cpu_cache:
            del flows

        # tail
        feats['decoder2'] = Forward(feats['decoder2'], self.tail, self.cpu_cache)

        outputs = []
        for i in range(0, t):
            feature_i = feats['decoder2'].pop(0)
            if self.cpu_cache:
                feature_i = feature_i.cuda()
            out = feature_i + lqs[:, i, :, :, :]
            if self.cpu_cache:
                out = out.cpu()
                torch.cuda.empty_cache()
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)

        if self.patch_test and (h_input, w_input) != (256, 256):
            outputs = outputs[:, :, :, :256, :256]
            outputs = rearrange(outputs, '(n h w) t c p1 p2 -> n t c (h p1) (w p2)', h=h_pad // 256, w=w_pad // 256)

        return outputs[:, :, :, :h_input, :w_input]

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.
        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        # if isinstance(pretrained, str):
        #     logger = get_root_logger()
        #     load_checkpoint(self, pretrained, strict=strict, logger=logger)
        # elif pretrained is not None:
        #     raise TypeError(f'"pretrained" must be a str or None. '
        #                     f'But received {type(pretrained)}.')
        pass



