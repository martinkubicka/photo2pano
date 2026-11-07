import numpy as np
import pytorch360convert
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

BACKBONE = "timm/vit_pe_spatial_tiny_patch16_512.fb"


def sinusoidal_pe(n, dim, device, dtype):
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) *
                    (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim))
    idx = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    pe = torch.zeros(n, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(idx * div)
    pe[:, 1::2] = torch.cos(idx * div)

    return pe


def mlp_head(dim, bins):
    return nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, bins))


class OrientationEstimator(nn.Module):
    """Photo2Pano: photo + 360 deg panorama -> logits over yaw, pitch, roll (and optionally FoV) bins.

    The panorama is given as a path or file-like object of the 4096x2048 canvas and is resampled into
    num_rows x num_cols perspective tiles. Photo and tiles share the ViT encoder; their tokens interact
    in one cross-attention layer shared by both directions. `query_black_mask` (B, Nq) marks the photo
    tokens that overlap the letterbox padding; they are excluded from the keys of the
    panorama -> photo attention.
    """

    def __init__(self, device, num_rows=4, num_cols=8, fov=(45, 45), tile_size=(512, 512),
                 fov_head=False, tf=None):
        super().__init__()
        self.transform = tf
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fov = fov
        self.tile_size = tile_size

        self.pitch_bins = 180
        self.yaw_bins = 360
        self.roll_bins = 360
        self.fov_bins = 180

        self.backbone = AutoModel.from_pretrained(BACKBONE).to(device)
        self.embed_dim = self.backbone.config.num_features
        d = self.embed_dim

        self.cross_attn = nn.MultiheadAttention(embed_dim=d, num_heads=8, batch_first=True)
        self.attn_pool = nn.Linear(d, 1)

        self.norm_pano = nn.LayerNorm(d)
        self.norm_query = nn.LayerNorm(d)
        self.norm_post_pano = nn.LayerNorm(d)
        self.norm_post_query = nn.LayerNorm(d)
        self.norm_fused = nn.LayerNorm(2 * d)

        self.pos_embed = nn.Parameter(torch.zeros(num_rows * num_cols, d))

        self.head_pitch = mlp_head(d, self.pitch_bins)
        self.head_yaw = mlp_head(d, self.yaw_bins)
        self.head_roll = mlp_head(d, self.roll_bins)
        self.head_fov = mlp_head(d, self.fov_bins) if fov_head else None

        j = torch.arange(num_rows, dtype=torch.float32)
        self.pitch_vals = 90.0 - (j + 0.5) * (180.0 / num_rows)
        i = torch.arange(num_cols, dtype=torch.float32)
        self.yaw_vals = -180.0 + (i + 0.5) * (360.0 / num_cols)

    def tiles(self, pano, device):
        """Perspective tiles of one panorama canvas, (num_rows * num_cols, 3, H, W) in [0, 1]."""
        e_img = np.array(Image.open(pano).convert("RGB"))
        e_img = torch.from_numpy(e_img).permute(2, 0, 1).float().div(255.0).to(device)
        out = []
        for pv in self.pitch_vals.tolist():
            for yv in self.yaw_vals.tolist():
                out.append(pytorch360convert.e2p(e_img, self.fov, float(yv), float(pv),
                                                 (self.tile_size[1], self.tile_size[0])))

        return torch.stack(out, 0)

    def forward(self, query_img, pano, query_black_mask=None, return_attention=False):
        B = len(pano)
        n_tiles = self.num_rows * self.num_cols
        device = query_img.device

        tiles = torch.cat([self.tiles(p, device) for p in pano], 0)
        pano_tokens = self.backbone((tiles - 0.5) / 0.5).last_hidden_state[:, 1:, :]
        Np = pano_tokens.size(1)
        pano_seq = pano_tokens.view(B, n_tiles, Np, self.embed_dim)
        pano_seq = pano_seq + self.pos_embed.unsqueeze(0).unsqueeze(2)
        pano_seq = pano_seq + sinusoidal_pe(Np, self.embed_dim, device, pano_seq.dtype).unsqueeze(0).unsqueeze(1)
        P = pano_seq.view(B, n_tiles * Np, self.embed_dim)

        Q = self.backbone(query_img).last_hidden_state[:, 1:, :]
        Nq = Q.size(1)
        Q = Q + sinusoidal_pe(Nq, self.embed_dim, device, Q.dtype).unsqueeze(0)

        if query_black_mask is None:
            mask = torch.zeros(B, Nq, dtype=torch.bool, device=device)
        else:
            mask = query_black_mask.to(device).bool()

        P = self.norm_pano(P)
        Q = self.norm_query(Q)
        P_ca, attn_pq = self.cross_attn(P, Q, Q, key_padding_mask=mask)
        Q_ca, attn_qp = self.cross_attn(Q, P, P)
        P = self.norm_post_pano(P + P_ca)
        Q = self.norm_post_query(Q + Q_ca)

        w_p = F.softmax(self.attn_pool(P).squeeze(-1), dim=1).unsqueeze(-1)
        w_q = F.softmax(self.attn_pool(Q).squeeze(-1), dim=1).unsqueeze(-1)
        z = self.norm_fused(torch.cat([(w_p * P).sum(1), (w_q * Q).sum(1)], dim=-1))

        out = [self.head_pitch(z), self.head_yaw(z), self.head_roll(z)]
        if self.head_fov is not None:
            out.append(self.head_fov(z))
        if return_attention:
            out.append({"pano_to_photo": attn_pq, "photo_to_pano": attn_qp, "pool_photo": w_q.squeeze(-1),
                        "mask": mask, "Np": Np, "Nq": Nq})

        return tuple(out)
