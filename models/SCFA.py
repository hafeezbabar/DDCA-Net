import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return self.alpha * torch.tanh(x) + (1-self.alpha) * F.hardswish(x)

class EnhancedDepthChannelAtt2(nn.Module):
    def __init__(self, dim, kernel=3) -> None:
        super().__init__()
        self.kernel = (1, kernel)
        pad_r = pad_l = kernel // 2
        self.pad = nn.ReflectionPad2d((pad_r, pad_l, 0, 0))
        
        # Dynamic channel grouping
        self.groups = max(1, dim // 16)
        self.conv = nn.Conv2d(dim, kernel*dim, kernel_size=1, 
                            stride=1, bias=False, groups=self.groups)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Replacing sequential activation with ParallelActivation
        self.filter_act = ParallelActivation()
        
        self.filter_bn = nn.GroupNorm(self.groups, kernel*dim)
        self.temp = nn.Parameter(torch.ones(1) * 0.05)
        
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        filter = self.conv(self.gap(x))
        filter = self.filter_bn(filter)
        # Using parallel activation
        filter = self.filter_act(filter) * self.temp.exp()
        
        b, c, h, w = filter.shape
        filter = filter.view(b, self.kernel[1], c//self.kernel[1], h*w)
        filter = filter.permute(0, 1, 3, 2).contiguous()
        
        B, C, H, W = x.shape
        out = x.permute(0, 2, 3, 1).view(B, H*W, C).unsqueeze(1)
        out = F.unfold(self.pad(out), kernel_size=self.kernel, stride=1)
        out = out.view(B, self.kernel[1], H*W, -1)
        
        out = torch.sum(out * filter, dim=1, keepdim=True)
        out = out.permute(0,3,1,2).reshape(B,C,H,W)
        
        return out * self.gamma + x * self.beta.



import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDecomposition(nn.Module):
    def __init__(self, features):
        super(FrequencyDecomposition, self).__init__()
        self.group_channels = features // 4
        
        # Frequency splits using group convolutions
        self.C_llf = nn.Sequential(
            nn.Conv2d(features, self.group_channels,
                      kernel_size=3, stride=1, padding=4, dilation=4,
                      groups=self.group_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.C_lf = nn.Sequential(
            nn.Conv2d(features, self.group_channels,
                      kernel_size=3, stride=1, padding=3, dilation=3,
                      groups=self.group_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.C_mf = nn.Sequential(
            nn.Conv2d(features, self.group_channels,
                      kernel_size=3, stride=1, padding=2, dilation=2,
                      groups=self.group_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.C_hf = nn.Sequential(
            nn.Conv2d(features, self.group_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1,
                      groups=self.group_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.R = nn.GELU()

    def forward(self, x):
        llf = self.R(self.C_llf(x))          
        lf = self.R(self.C_lf(x) - llf)     
        mf = self.R(self.C_mf(x) - lf)      
        hf = self.R(self.C_hf(x) - mf)      
        return torch.cat((llf, lf, mf, hf), dim=1)

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)

        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class ODConv2d2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d2, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        
        # Add frequency decomposition
        self.freq_decomp = FrequencyDecomposition(in_planes)
        
        # Attention now takes concatenated frequency features as input
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                 reduction=reduction, kernel_num=kernel_num)
                                 
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                 requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # First apply frequency decomposition
        freq_features = self.freq_decomp(x)
        #print("shape of freq_features", freq_features.shape)
        
        # Get attention weights using frequency features
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(freq_features)
        
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
            
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                         dilation=self.dilation, groups=self.groups * batch_size)
                         
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        # First apply frequency decomposition
        freq_features = self.freq_decomp(x)
        
        #print("shape of freq_features", freq_features.shape)
        # Get attention weights using frequency features
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(freq_features)
        
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                         dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

def get_temperature(iteration, epoch, iter_per_epoch, temp_epoch=10, temp_init=30.0):
    total_temp_iter = iter_per_epoch * temp_epoch
    current_iter = iteration + epoch * iter_per_epoch
    temperature = 1.0 + max(0, (temp_init - 1.0) * ((total_temp_iter - current_iter) / total_temp_iter))
    return temperature



class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.dyna_ch = EnhancedDepthChannelAtt2(in_channel) if filter else nn.Identity()
        self.odconv = ODConv2d2(in_channel, in_channel, kernel_size=3, padding=1) if filter else nn.Identity()
    
        self.proj = nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel)
        self.proj_act = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.proj(out)
        out = self.proj_act(out)
        out = self.dyna_ch(out)
        out = self.odconv(out)
        out = self.conv2(out)
        
        return out + x
