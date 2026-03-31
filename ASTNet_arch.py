import math
import numbers

import torch
from einops import rearrange, repeat
# from timm.layers import trunc_normal_
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from basicsr.archs.kpn_pixel import IDynamicDWConv


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DownSampling(nn.Module):
    """
    PixelUnshuffle 下采样两倍
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownSampling, self).__init__()

        self.conv = nn.Conv2d(in_channels * 2 * 2, out_channels, kernel_size=1)
        self.down = nn.PixelUnshuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.conv(x)
        return x


class UpSampling(nn.Module):
    """
    PixelUnshuffle 上采样两倍
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpSampling, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * 2 * 2, kernel_size=1)
        self.down = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.down(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResBlock, self).__init__()

        self.conv_in = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.conv_in(x)
        x = self.act(x)
        x = self.conv_out(x)

        return x + res


class VSA(nn.Module):
    """
    垂直空间增强
    """

    def __init__(self, channels: int) -> None:
        super(VSA, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d((None, 1))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        w = self.avg(x)
        w = self.conv(w)
        w = self.act(w)
        x = x * w

        return x


class HSA(nn.Module):
    """
    水平空间增强
    """

    def __init__(self, channels: int) -> None:
        super(HSA, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        w = self.avg(x)
        w = self.conv(w)
        w = self.act(w)
        x = x * w

        return x


class HWT(nn.Module):
    def __init__(self, channel_in):
        super(HWT, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out[:, :self.channel_in, :, :], out[:, self.channel_in:self.channel_in * 4, :, :]
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)


def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # print(windows.shape)
    # B' ,Wh ,Ww ,C
    # B = int(windows.shape[0] / (H * W / win_size / win_size))
    B = windows.shape[0]
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SWSA(nn.Module):
    """
    Sparse Window Self-Attention
    """

    def __init__(self, channels, win_size, num_heads, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = channels
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # trunc_normal_(self.relative_position_bias_table, std=.02)

        self.conv_in = nn.Conv3d(channels, channels * 3, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_out = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.dconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3)

        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.dconv(x)
        x = rearrange(x, 'N C H W -> N H W C')

        x = window_partition(x, win_size=self.win_size[0])
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = rearrange(q, 'N H W (head C) -> N head (H W) C', head=self.num_heads, C=C // self.num_heads)
        k = rearrange(k, 'N H W (head C) -> N head (H W) C', head=self.num_heads, C=C // self.num_heads)
        v = rearrange(v, 'N H W (head C) -> N head (H W) C', head=self.num_heads, C=C // self.num_heads)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh * Ww,Wh * Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh * Ww, Wh * Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        attn0 = self.softmax(attn)
        attn1 = self.relu(attn) ** 2

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * 0.25 + attn1 * 0.75
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * T, H * W, C)
        x = window_reverse(x, self.win_size[0], H, W)

        x = rearrange(x, '(B T) H W C -> B C T H W', H=H, W=W, B=B)
        x = self.conv_out(x)
        x = rearrange(x, 'B C T H W -> B T C H W')

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, HW):
        flops = 0
        flops += HW * self.in_channels * self.kernel_size ** 2 / self.stride ** 2
        flops += HW * self.in_channels * self.out_channels
        print("SeqConv2d:{%.2f}" % (flops / 1e9))
        return flops


class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops


class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        # trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H * W, H * W)

        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops


# Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, t, c, h, w = x.shape

        x = rearrange(x, 'B T C H W -> (B T) C H W', B=b, T=t)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = rearrange(out, '(B T) C H W -> B T C H W', B=b, T=t)
        return out


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


class SCSA(nn.Module):
    """
    Sparse Channel Self-Attention (STGSA)
    """

    def __init__(self, channels: int, head: int) -> None:
        super().__init__()

        self.head = head
        self.conv_in = nn.Conv3d(channels, channels * 3, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_out = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.dconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3)

        self.beta = nn.Parameter(torch.ones(head, 1, 1))
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.dconv(x)
        q, k, v = torch.chunk(x, 3, dim=1)

        q = rearrange(q, 'N (c head) H W -> N head c (H W)', head=self.head)
        k = rearrange(k, 'N (c head) H W -> N head c (H W)', head=self.head)
        v = rearrange(v, 'N (c head) H W -> N head c (H W)', head=self.head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.beta
        attn_1 = attn.softmax(dim=-1)

        w1 = torch.exp(self.weight[0]) / torch.sum(torch.exp(self.weight))
        w2 = torch.exp(self.weight[1]) / torch.sum(torch.exp(self.weight))
        attn_2 = torch.relu(attn) ** 2
        attn = attn_1 * w1 + attn_2 * w2
        out = (attn @ v)

        out = rearrange(out, 'N head c (H W) -> N (head c) H W', head=self.head, H=H, W=W)

        out = rearrange(out, '(B T) C H W -> B C T H W', B=B, T=T)
        out = self.conv_out(out)
        out = rearrange(out, 'B C T H W -> B T C H W')

        return out


class MLP(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.LeakyReLU(),
            nn.Linear(channels * 2, channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> (B T) H W C')
        x = self.body(x)
        x = rearrange(x, '(B T) H W C -> B T C H W', B=B, T=T)

        return x


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> (B T) (H W) C')
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)
        x = rearrange(x, '(B T) (H W) C -> B T C H W', B=B, T=T, H=H, W=W)

        return x


class GDFN(nn.Module):
    def __init__(self, channels: int) -> None:
        super(GDFN, self).__init__()
        hidden_channels = int(channels * 2.66)
        self.conv_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        self.conv_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)

        self.dconv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.dconv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)

        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> (B T) C H W')
        x = self.conv_in(x)
        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = self.dconv1(x1)
        x1 = self.act(x1)
        x2 = self.dconv2(x2)
        x = x1 * x2

        x = self.conv_out(x)
        x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)

        return x


class DGFFN(nn.Module):
    """
    Bidirectional Feature-Enhanced Feed-Forward Networks (BFEFN)
    """

    def __init__(self, channels: int) -> None:
        super(DGFFN, self).__init__()
        hidden_channels = int(channels * 2.66)
        self.conv_in = nn.Conv3d(channels, hidden_channels * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_out = nn.Conv3d(hidden_channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.dconv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.dconv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)

        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = self.dconv1(x1)
        x1 = self.act(x1)
        x2 = self.dconv2(x2)
        x = x1 * x2

        x = rearrange(x, '(B T) C H W -> B C T H W', B=B, T=T)
        x = self.conv_out(x)
        x = rearrange(x, 'B C T H W -> B T C H W')

        return x


class DilatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DilatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, t, c, h, w = x.shape

        x = rearrange(x, 'B T C H W -> (B T) C H W')
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = rearrange(out, '(B T) C H W -> B T C H W', B=b, T=t)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, head: int, win_size) -> None:
        super(TransformerBlock, self).__init__()

        self.norm = LayerNorm(channels, LayerNorm_type='WithBias')
        self.attn1 = SWSA(channels, win_size=win_size, num_heads=head)
        self.attn2 = DilatedMDTA(channels, head, False)
        self.ffn = GDFN(channels)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        res = x
        x = rearrange(x, 'B T C H W -> (B T) C H W')
        x = self.norm(x)
        x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)
        x = self.attn1(x)
        x = x + res

        res = x
        x = rearrange(x, 'B T C H W -> (B T) C H W')
        x = self.norm(x)
        x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)
        x = self.attn2(x)
        x = x + res

        res = x
        x = rearrange(x, 'B T C H W -> (B T) C H W')
        x = self.norm(x)
        x = rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)
        x = self.ffn(x)
        x = x + res

        return x


class ForwardFeatureFusion(nn.Module):
    """
    前向特征融合
    """

    def __init__(self, channels: int, num_blocks: int) -> None:
        super(ForwardFeatureFusion, self).__init__()

        self.conv1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=1)

        self.dynamic_conv = IDynamicDWConv(channels, kernel_size=3, group_channels=1, down=4, conv_group=1)
        self.res_layer = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        feat_list = []
        forward_feat = x.new_zeros((B, C, H, W), requires_grad=True)
        for i in range(T):
            feat = x[:, i, ...]
            feat_fusion = torch.cat((feat, forward_feat), dim=1)
            feat_fusion = self.conv1(feat_fusion)
            feat1, feat2 = torch.chunk(feat_fusion, 2, dim=1)
            feat_fusion = feat1 * feat2
            feat_fusion = feat * torch.sigmoid(feat_fusion) + forward_feat * (1 - torch.sigmoid(feat_fusion))
            feat_fusion = self.conv2(feat_fusion)
            feat_fusion = self.dynamic_conv(feat_fusion)
            feat_fusion = self.res_layer(feat_fusion)
            feat_list.append(feat_fusion)

        out = torch.stack(feat_list, dim=1)
        return out


class BackwardFeatureFusion(nn.Module):
    """
    后向特征融合channels, kernel_size=3, group_channels=1, down=4, conv_group=1
    """

    def __init__(self, channels: int, num_blocks: int) -> None:
        super(BackwardFeatureFusion, self).__init__()

        self.conv1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=1)

        self.dynamic_conv = IDynamicDWConv(channels, kernel_size=3, group_channels=1, down=4, conv_group=1)
        self.res_layer = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        feat_list = []
        forward_feat = x.new_zeros((B, C, H, W), requires_grad=True)
        for i in range(T - 1, -1, -1):
            feat = x[:, i, ...]
            feat_fusion = torch.cat((feat, forward_feat), dim=1)
            feat_fusion = self.conv1(feat_fusion)
            feat1, feat2 = torch.chunk(feat_fusion, 2, dim=1)
            feat_fusion = feat1 * feat2
            feat_fusion = feat * torch.sigmoid(feat_fusion) + forward_feat * (1 - torch.sigmoid(feat_fusion))
            feat_fusion = self.conv2(feat_fusion)
            feat_fusion = self.dynamic_conv(feat_fusion)
            feat_fusion = self.res_layer(feat_fusion)
            feat_list.append(feat_fusion)

        feat_list.reverse()
        out = torch.stack(feat_list, dim=1)
        return out


class Transformer(nn.Module):
    def __init__(self, channels: int, transformer_blocks: list, heads: list, win_size: list) -> None:
        super(Transformer, self).__init__()

        self.Encoder = nn.Sequential(
            *[TransformerBlock(channels, heads[0], win_size=win_size) for _ in range(transformer_blocks[0])]
        )
        self.latent = nn.Sequential(
            *[TransformerBlock(channels * 2, heads[1], win_size=win_size) for _ in range(transformer_blocks[1])]
        )
        self.Decoder = nn.Sequential(
            *[TransformerBlock(channels, heads[0], win_size=win_size) for _ in range(transformer_blocks[0])]
        )
        self.down = DownSampling(in_channels=channels, out_channels=channels * 2)
        self.up = UpSampling(in_channels=channels * 2, out_channels=channels)
        self.resconv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape
        res = x
        x = self.Encoder(x)
        res1 = x
        x = rearrange(self.down(rearrange(x, 'B T C H W -> (B T) C H W')), '(B T) C H W -> B T C H W', B=B)
        x = rearrange(self.latent(x), 'B T C H W -> (B T) C H W')
        x = self.up(x)
        x = rearrange(self.resconv(torch.cat((x, rearrange(res1, 'B T C H W -> (B T) C H W')), dim=1)),
                      '(B T) C H W -> B T C H W', B=B)
        x = self.Decoder(x)

        return x + res


class ASTNet_arch(nn.Module):
    def __init__(self, channels: int, transformer_list: list, head_list: list, win_size: list, fusion_blocks: int,
                 refinement_blocks: int) -> None:
        super().__init__()

        fusion_channels = channels + channels // 2
        self.conv_in = nn.Conv3d(3, channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.conv_out = nn.Conv3d(channels, 3, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)

        self.WT = HWT(channels)
        self.conv_scale1 = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, groups=3),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, groups=3)
        )
        self.conv_scale2 = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, groups=3),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, groups=3)
        )

        self.Transformer = Transformer(channels, transformer_list, head_list, win_size)
        self.Backward = BackwardFeatureFusion(fusion_channels, fusion_blocks)
        self.Forward = ForwardFeatureFusion(fusion_channels, fusion_blocks)
        if refinement_blocks > 0:
            self.Refinement = nn.Sequential(
                *[ResBlock(channels) for _ in range(refinement_blocks)]
            )
        else:
            self.Refinement = None

        self.channel_up = nn.Conv2d(channels, fusion_channels, kernel_size=1)
        self.channel_down = nn.Conv2d(fusion_channels, channels, kernel_size=1)

        self.maps = None

    def forward(self, x: Tensor) -> Tensor:
        res0 = x

        B, T, C, H, W = x.shape
        x = rearrange(x, 'B T C H W -> B C T H W')
        x = self.conv_in(x)
        x = rearrange(x, 'B C T H W -> (B T) C H W')

        low1, high1 = self.WT(x)
        low2, high2 = self.WT(low1)
        high2 = self.conv_scale2(high2)
        low2 = rearrange(low2, '(B T) C H W -> B T C H W', B=B, T=T)

        low2 = rearrange(self.Transformer(low2), 'B T C H W -> (B T) C H W')

        low1 = torch.cat((low2, high2), dim=1)
        low1 = self.WT(low1, rev=True)

        low1 = self.channel_up(low1)
        low1 = rearrange(low1, '(B T) C H W -> B T C H W', B=B, T=T)
        low1 = self.Backward(low1)
        low1 = self.Forward(low1)
        low1 = rearrange(low1, 'B T C H W -> (B T) C H W')
        low1 = self.channel_down(low1)

        high1 = self.conv_scale1(high1)
        low1 = torch.cat((low1, high1), dim=1)
        x = self.WT(low1, rev=True)

        if self.Refinement is not None:
            x = self.Refinement(x)

        x = rearrange(x, '(B T) C H W -> B C T H W', B=B, T=T)
        x = self.conv_out(x)
        x = rearrange(x, 'B C T H W -> B T C H W')

        return x.contiguous() + res0
