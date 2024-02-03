import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_corr(z, x):
    """Pixel-wise correlation (implementation by for-loop and convolution)
    The speed is slower because the for-loop"""
    size = z.size()  # (bs, c, hz, wz)
    CORR = []
    for i in range(len(x)):
        ker = z[i:i + 1]  # (1, c, hz, wz)
        fea = x[i:i + 1]  # (1, c, hx, wx)
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)  # (hz * wz, c)
        ker = ker.unsqueeze(2).unsqueeze(3)  # (hz * wz, c, 1, 1)
        co = F.conv2d(fea, ker.contiguous())  # (1, hz * wz, hx, wx)
        CORR.append(co)
    corr = torch.cat(CORR, 0)  # (bs, hz * wz, hx, wx)
    return corr


def pixel_corr_mat(z, x):
    """Pixel-wise correlation (implementation by matrix multiplication)
    The speed is faster because the computation is vectorized"""
    b, c, h, w = x.size()
    z_mat = z.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)
    x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
    return torch.matmul(z_mat, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)


class CAModule(nn.Module):
    """Channel attention module"""

    def __init__(self, channels=64, reduction=1):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h

    
class PWCoordAttention(nn.Module):
    """
    Pointwise Correlation & CoordAttention
    """

    def __init__(self, num_channel, cat=False, CA=True, matrix=False):
        super(PWCoordAttention, self).__init__()
        self.cat = cat
        self.CA = CA
        self.matrix = matrix
        if self.CA:
            self.CA_layer = CoordAttention(in_channels=num_channel, out_channels=num_channel)

    def forward(self, z, x):
        z11 = z
        x11 = x
        # pixel-wise correlation
        if self.matrix:
            corr = pixel_corr_mat(z11, x11)
        else:
            corr = pixel_corr(z11, x11)
#         print(corr.shape)
        if self.CA:
            # channel attention
            opt = self.CA_layer(corr)
            if self.cat:
                return torch.cat([opt, x11], dim=1)
            else:
                return opt
        else:
            return corr

class PWCA(nn.Module):
    """
    Pointwise Correlation & Channel Attention
    """

    def __init__(self, num_channel, cat=False, CA=True, matrix=False):
        super(PWCA, self).__init__()
        self.cat = cat
        self.CA = CA
        self.matrix = matrix
        if self.CA:
            self.CA_layer = CAModule(channels=num_channel)

    def forward(self, z, x):
        z11 = z
        x11 = x
        # pixel-wise correlation
        if self.matrix:
            corr = pixel_corr_mat(z11, x11)
        else:
            corr = pixel_corr(z11, x11)
        if self.CA:
            # channel attention
            opt = self.CA_layer(corr)
            if self.cat:
                return torch.cat([opt, x11], dim=1)
            else:
                return opt
        else:
            return corr

class Point_Neck(nn.Module):
    def __init__(self, num_kernel = 64, cat=False, matrix=True, adjust=True, corr_channel=64, adj_channel=128):
        super(Point_Neck, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = PWCA(num_kernel, cat=cat, CA=True, matrix=matrix)
        self.adj_layer = nn.Conv2d(corr_channel, adj_channel, 1)

    def forward(self, kernel, search, stride_idx=None):
        corr_feat = self.pw_corr(kernel, search)
        if self.adjust:
            corr_feat = self.adj_layer(corr_feat)
        return corr_feat

    
class Point_Neck_CoordAttention(nn.Module):
    def __init__(self, num_kernel = 64, cat=False, matrix=True, adjust=True, corr_channel=64, adj_channel=128):
        super(Point_Neck_CoordAttention, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = PWCoordAttention(num_kernel, cat=cat, CA=True, matrix=matrix)
        self.adj_layer = nn.Conv2d(corr_channel, adj_channel, 1)

    def forward(self, kernel, search, stride_idx=None):
        corr_feat = self.pw_corr(kernel, search)
        if self.adjust:
            corr_feat = self.adj_layer(corr_feat)
        return corr_feat

    
class FEAR_PWCA(nn.Module):
    """
    FEAR Pointwise Correlation & Channel Attention
    """

    def __init__(self, num_channel, cat=False, matrix=False):
        super(FEAR_PWCA, self).__init__()
        self.cat = cat
        self.matrix = matrix

    def forward(self, z, x):
        z11 = z
        x11 = x
        # pixel-wise correlation
        if self.matrix:
            corr = pixel_corr_mat(z11, x11)
        else:
            corr = pixel_corr(z11, x11)
        if self.cat:
            return torch.cat([corr, x11], dim=1)
        else:
            return corr

        
class Point_Neck_FEAR(nn.Module):
    def __init__(self, num_kernel = 64, cat=False, matrix=True, adjust=True, corr_channel=64, adj_channel=128):
        super(Point_Neck_FEAR, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = FEAR_PWCA(num_kernel, cat=cat, matrix=matrix)
        self.adj_layer = nn.Conv2d(corr_channel, adj_channel, 1)

    def forward(self, kernel, search, stride_idx=None):
        corr_feat = self.pw_corr(kernel, search)
        if self.adjust:
            corr_feat = self.adj_layer(corr_feat)
        return corr_feat

	
class LightTrack_PWCA(nn.Module):
    """
    Pointwise Correlation & Channel Attention
    """

    def __init__(self, num_channel, cat=False, CA=True, matrix=False):
        super(LightTrack_PWCA, self).__init__()
        self.cat = cat
        self.CA = CA
        self.matrix = matrix
        if self.CA:
            self.CA_layer = CAModule(channels=num_channel)

    def forward(self, z, x):
        z11 = z
        x11 = x
        # pixel-wise correlation
        if self.matrix:
            corr = pixel_corr_mat(z11, x11)
        else:
            corr = pixel_corr(z11, x11)
        if self.CA:
            # channel attention
            opt = self.CA_layer(corr)
            if self.cat:
                return torch.cat([opt, x11], dim=1)
            else:
                return opt
        else:
            return corr

		
class Point_Neck_LightTrack(nn.Module):
    def __init__(self, num_kernel = 64, cat=False, matrix=True, adjust=True, corr_channel=64, adj_channel=128):
        super(Point_Neck_LightTrack, self).__init__()
        self.adjust = adjust
        '''Point-wise Correlation & Adjust Layer (unify the num of channels)'''
        self.pw_corr = LightTrack_PWCA(num_kernel, cat=cat, matrix=matrix)
        self.adj_layer = nn.Conv2d(corr_channel, adj_channel, 1)

    def forward(self, kernel, search, stride_idx=None):
        corr_feat = self.pw_corr(kernel, search)
        if self.adjust:
            corr_feat = self.adj_layer(corr_feat)
        return corr_feat

    
def pixel_corr_mat_2(z, x):
    """Pixel-wise correlation (implementation by matrix multiplication)
    The speed is faster because the computation is vectorized"""
    b, c, h, w = x.size()
    x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
    return torch.matmul(z, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)    

    
class Hierarchy_Point_Neck(nn.Module):
    def __init__(self, num_kernel = 64, cat=False, adjust=True, corr_channel=64, adj_channel=128, hierarchy = 1):
        super(Hierarchy_Point_Neck, self).__init__()
        self.adjust = adjust
        self.hierarchy = hierarchy
        self.CA_layer = CAModule(channels = num_kernel*hierarchy)
        
        self.cat = cat
        self.CA = True
        
        self.adj_layer = nn.Conv2d(corr_channel, adj_channel, 1)
    
    def forward(self, kernel, search):
        z = kernel
        x = search

        b, c, h, w = x.size()
        z_mat = z.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)

        for i in range(self.hierarchy):

            x = pixel_corr_mat_2(z_mat, x)
            ans = torch.cat([x, ans], dim=1) if i > 0 else x  

            x = torch.cat([x, x], dim=1)

        if self.CA:
            # channel attention
            opt = self.CA_layer(ans)
            if self.cat:
                corr_feat = torch.cat([opt, search], dim=1)
            else:
                corr_feat = opt
        else:
            corr_feat = ans
        
        if self.adjust:
            corr_feat = self.adj_layer(corr_feat)
        return corr_feat
    
