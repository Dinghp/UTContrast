import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class AuxiliaryHead(nn.Module):
    def __init__(self, in_c,hidden_c,out_c):
        super(AuxiliaryHead, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, hidden_c, kernel_size=3, padding=1),
            CBR(hidden_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

        self.cbr_uc = nn.Sequential(
            CBR(in_c, hidden_c, kernel_size=3, padding=1),
            CBR(hidden_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.branch_fg = nn.Sequential(
            CBR(out_c, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 9, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

        # self.branch_bg = nn.Sequential(
        #     CBR(out_c, 64, kernel_size=3, padding=1),
        #     nn.Conv2d(64, 1, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )
        self.branch_uc = nn.Sequential(
            CBR(out_c, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

    def forward(self,x):
        f_fg, f_uc=self.cbr_fg(x),self.cbr_uc(x)
        mask_fg = self.branch_fg(f_fg)

        mask_uc = self.branch_uc(f_uc)
        return mask_fg,  mask_uc

class AuxiliaryHead_acdc(nn.Module):
    def __init__(self, in_c,hidden_c,out_c):
        super(AuxiliaryHead_acdc, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, hidden_c, kernel_size=3, padding=1),
            CBR(hidden_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

        self.cbr_uc = nn.Sequential(
            CBR(in_c, hidden_c, kernel_size=3, padding=1),
            CBR(hidden_c, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.branch_fg = nn.Sequential(
            CBR(out_c, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 4, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

        # self.branch_bg = nn.Sequential(
        #     CBR(out_c, 64, kernel_size=3, padding=1),
        #     nn.Conv2d(64, 1, kernel_size=1, padding=0),
        #     # nn.Sigmoid()
        # )
        self.branch_uc = nn.Sequential(
            CBR(out_c, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

    def forward(self,x):
        f_fg, f_uc=self.cbr_fg(x),self.cbr_uc(x)
        mask_fg = self.branch_fg(f_fg)

        mask_uc = self.branch_uc(f_uc)
        return mask_fg,  mask_uc
