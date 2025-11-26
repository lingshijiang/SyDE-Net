import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    """Generic 2D/3D convolution block with optional BN and LeakyReLU."""

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super().__init__()
        self.relu = relu
        self.use_bn = bn

        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x


class CLC(nn.Module):
    """Conv-LeakyReLU-Conv block used inside FG_RCA."""

    def __init__(self, in_ch, out_ch=None, k=3, negative_slope=0.1):
        super().__init__()
        out_ch = out_ch or in_ch
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def _choose_gn_groups(channels: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class DNRU(nn.Module):
    """Depthwise block with optional upsampling used by FG_RCA."""

    def __init__(self, channels, up_scale=1):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(channels), channels)
        self.relu = nn.ReLU(inplace=True)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.dwconv(x)
        x = self.gn(x)
        x = self.relu(x)
        if self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x


def vector_to_grid(x_vec):
    """(B, C, 1, 1) -> (B, 1, Hc, Wc) for FFT processing."""
    B, C, _, _ = x_vec.shape
    Hc = int(math.floor(math.sqrt(C)))
    Wc = int(math.ceil(C / Hc))
    pad = Hc * Wc - C
    if pad > 0:
        x_vec = F.pad(x_vec.view(B, C), (0, pad))
    else:
        x_vec = x_vec.view(B, C)
    grid = x_vec.view(B, 1, Hc, Wc)
    return grid, (Hc, Wc, C, pad)


def grid_to_vector(grid, meta):
    """Inverse of vector_to_grid."""
    Hc, Wc, C, pad = meta
    vec = grid.view(grid.size(0), Hc * Wc)
    if pad > 0:
        vec = vec[:, :C]
    return vec.view(grid.size(0), C, 1, 1)


def normalized_freq_radius(h, w, device=None, dtype=None):
    fy = torch.fft.fftfreq(h, d=1.0).to(device=device, dtype=dtype)
    fx = torch.fft.fftfreq(w, d=1.0).to(device=device, dtype=dtype)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(fx * fx + fy * fy)
    r_max = math.sqrt((0.5 ** 2) * 2.0)
    return (r / r_max).clamp(0, 1)


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        y = self.conv1d(y).transpose(1, 2).unsqueeze(-1)
        return torch.sigmoid(y)


class FG_RCA(nn.Module):
    """Fourier-Gated Residual Channel Attention block used for mono SE."""

    def __init__(
        self,
        channels: int,
        up_scale: int = 1,
        negative_slope: float = 0.1,
        eca_ksize: int = 3,
        freq_gate_init_t: float = 0.35,
        freq_gate_init_s: float = 8.0,
    ):
        super().__init__()
        self.channels = channels
        self.spatial_pre = CLC(channels, channels, k=3, negative_slope=negative_slope)
        act = nn.GELU()
        self.amp_mlp = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, bias=False), act, nn.Conv2d(1, 1, kernel_size=1, bias=False))
        self.pha_mlp = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, bias=False), act, nn.Conv2d(1, 1, kernel_size=1, bias=False))
        self.freq_t = nn.Parameter(torch.tensor(freq_gate_init_t, dtype=torch.float32))
        self.freq_s = nn.Parameter(torch.tensor(freq_gate_init_s, dtype=torch.float32))
        self.hi_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lo_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.phs_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.eca = ECA(channels, k_size=eca_ksize)
        self.mix_alpha = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))
        self.mix_beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.post = DNRU(channels, up_scale=up_scale)

    @staticmethod
    def _sigm(x):
        return torch.sigmoid(x)

    def _freq_gate(self, h, w, device, dtype):
        r = normalized_freq_radius(h, w, device=device, dtype=dtype)
        gate = torch.sigmoid(self.freq_s * (r - self.freq_t))
        scale = self.lo_gain * (1.0 - gate) + self.hi_gain * gate
        return gate, scale

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels
        feat = self.spatial_pre(x)
        chan_desc = F.adaptive_avg_pool2d(feat, 1)
        grid, meta = vector_to_grid(chan_desc)
        spec = torch.fft.fft2(grid.float())
        amp = torch.abs(spec)
        pha = torch.angle(spec)
        _, Hc, Wc = spec.shape[1:]
        gate, scale = self._freq_gate(Hc, Wc, grid.device, grid.dtype)
        amp_adj = amp * (1.0 + self.amp_mlp(amp)) * scale.unsqueeze(0).unsqueeze(0)
        pha_adj = pha + self.phs_scale * gate.unsqueeze(0).unsqueeze(0) * torch.tanh(self.pha_mlp(pha))
        spec_new = torch.polar(amp_adj, pha_adj)
        grid_ifft = torch.fft.ifft2(spec_new).real
        weight_vec = grid_to_vector(grid_ifft, meta)
        w_freq = torch.sigmoid(weight_vec)
        w_eca = self.eca(feat)
        alpha = torch.sigmoid(self.mix_alpha)
        w = self._sigm(self.mix_beta) * (alpha * w_freq + (1 - alpha) * w_eca)
        y = feat * w
        out = y + x
        return self.post(out)


class BasicConv_IN(nn.Module):
    """Basic conv + InstanceNorm block used by MONSTER stems."""

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super().__init__()
        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x


class Conv2x(nn.Module):
    """Two-stage conv block with optional concatenation used in GRU decoder."""

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super().__init__()
        self.concat = concat
        self.is_3d = is_3d 

        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels * 2, out_channels * mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="nearest")
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        return self.conv2(x)


class Conv2x_IN(nn.Module):
    """Conv2x variant that uses BasicConv_IN."""

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super().__init__()
        self.concat = concat
        self.is_3d = is_3d 

        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels * 2, out_channels * mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="nearest")
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        return self.conv2(x)


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view(B, num_groups, channels_per_group, H, W).mean(dim=2)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    return volume.contiguous()


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device).view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, dim=1, keepdim=True)


class FeatureAtt(nn.Module):
    """Channel attention used on cost volumes."""

    def __init__(self, cv_chan, feat_chan):
        super().__init__()
        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
        )

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        return torch.sigmoid(feat_att) * cv


def context_upsample(disp_low, up_weights):
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * 4, w * 4), mode="nearest").reshape(b, 9, h * 4, w * 4)
    return (disp_unfold * up_weights).sum(1)

