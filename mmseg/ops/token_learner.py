from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProduct(nn.Module):
    def forward(self, x, y):
        return x * y


class Tokenlearner(nn.Module):
    def __init__(self, in_channels, num_tokens, num_conv=4) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_channels)
        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if i == 0 else num_tokens,
                        num_tokens,
                        kernel_size=(3, 3),
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(inplace=True) if i < num_conv - 1 else nn.Identity(),
                )
                for i in range(num_conv)
            ]
        )
        self.num_tokens = num_tokens
        self.dot = DotProduct()

    def forward(self, x):
        B, C, H, W = x.shape
        selected = x.permute(0, 2, 3, 1)
        selected = self.ln(selected)
        selected = selected.permute(0, 3, 1, 2)
        fmap = self.convs(selected)
        fmap = torch.sigmoid(fmap.reshape(fmap.shape[0], fmap.shape[1], -1))
        fmap = fmap.reshape(B, self.num_tokens, H, W).unsqueeze(2)
        # out = (x.unsqueeze(1) * fmap).mean(dim=(-2, -1))
        out = self.dot(x.unsqueeze(1), fmap).mean(dim=(-2, -1))
        return out


class TokenlearnerV11(nn.Module):
    def __init__(self, in_channels, num_tokens, num_conv=2, dropout=0.) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_channels)
        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=(1, 1),
                        stride=1,
                        groups=8,
                        bias=False,
                    ),
                )
                for i in range(num_conv - 1)
            ],
            nn.Conv2d(
                in_channels,
                num_tokens,
                kernel_size=(1, 1),
                stride=1,
                bias=False,
            ),
        )
        self.out_convs = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, 1),
            stride=1,
            groups=8,
            bias=False,
        )
        self.num_tokens = num_tokens
        self.bmm = MatrixProduct()

    def forward(self, x):
        B, C, H, W = x.shape
        selected = x.permute(0, 2, 3, 1)
        selected = self.ln(selected)
        selected = selected.permute(0, 3, 1, 2)
        fmap = self.convs(selected)       # (B, S, H, W)
        fmap = fmap.view(B, -1, H * W)      # (B, S, N)
        fmap = torch.softmax(fmap, dim=-1)

        feats = self.out_convs(x)
        feats = feats.view(B, C, -1).permute(0, 2, 1)

        # out = torch.einsum('bsn,bnc->bsc', fmap, feats)
        out = self.bmm(fmap, feats)
        return out


class MatrixProduct(nn.Module):
    def forward(self, x, y):
        return torch.bmm(x, y)


class TokenFuser(nn.Module):
    def __init__(self, in_channels, num_tokens, dropout=0.) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)
        self.ln_o = nn.LayerNorm(in_channels)
        self.proj = nn.Linear(num_tokens, num_tokens)

        self.conv = nn.Conv2d(
            in_channels,
            num_tokens,
            kernel_size=(1, 1),
            stride=1,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.dot = DotProduct()

        
    def forward(self, x, x_origin):
        # x: (B, S, C) 
        # x_origin: (B, H, W, C)
        B, S, C = x.shape
        x = self.ln1(x)
        x = x.permute(0, 2, 1)  # (B, C, S)
        x = self.proj(x)    # (B, C, S)
        x = x.permute(0, 2, 1)  # (B, S, C)
        x = self.ln2(x)
        x = x[:, :, None, None]    # (B, S, 1, 1, C)

        q = self.ln_o(x_origin)
        q = q.permute(0, 3, 1, 2)
        q = self.conv(q)   # (B, S, H, W)
        q = torch.sigmoid(q)
        q = q.unsqueeze(-1)     # (B, S, H, W, 1)

        # out = torch.einsum('bns,bsc->bnc', q, x)
        out = self.dot(q, x)
        out = out.sum(1)    # (B, H, W, C)
        out = self.dropout(out) + x_origin
        
        return out 