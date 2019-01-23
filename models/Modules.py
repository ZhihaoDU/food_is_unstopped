import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveAttention(nn.Module):
    def __init__(self, input_nf, atten_nf, is_residual):
        super(NaiveAttention, self).__init__()
        self.g_filter = nn.Sequential(
            nn.Conv2d(input_nf, atten_nf, 1, 1, 0),
        )
        self.h_filter = nn.Sequential(
            nn.Conv2d(input_nf, atten_nf, 1, 1, 0),
        )
        self.is_residual = is_residual

    def forward(self, *input):
        x = input[0]
        g = self.g_filter(x)
        h = self.h_filter(x)
        bb, cc, hh, ww = g.size()
        g = g.view(bb, cc, hh*ww)
        h = h.view(bb, cc, hh*ww)
        alpha = F.softmax(torch.bmm(g.permute(0, 2, 1), h), dim=2)
        x = x.view(bb, x.size(1), hh*ww)
        r = torch.bmm(alpha, x.permute(0, 2, 1))
        r = r.permute(0, 2, 1).contiguous().view(bb, x.size(1), hh, ww)
        if self.is_residual:
            r = r + x
        return F.relu(r), alpha


class SelfAttention(nn.Module):
    def __init__(self, input_nf, atten_nf, is_residual):
        super(SelfAttention, self).__init__()
        self.g_filter = nn.Sequential(
            nn.Conv2d(input_nf, atten_nf, 1, 1, 0),
        )
        self.is_residual = is_residual

    def forward(self, *input):
        x = input[0]
        g = self.g_filter(x)
        bb, cc, hh, ww = g.size()
        g = g.view(bb, cc, hh*ww)
        alpha = F.softmax(torch.bmm(g.permute(0, 2, 1), g), dim=2)
        x = x.view(bb, x.size(1), hh*ww)
        r = torch.bmm(alpha, x.permute(0, 2, 1))
        r = r.permute(0, 2, 1).contiguous().view(bb, x.size(1), hh, ww)
        if self.is_residual:
            r = r + x
        return F.relu(r), alpha
