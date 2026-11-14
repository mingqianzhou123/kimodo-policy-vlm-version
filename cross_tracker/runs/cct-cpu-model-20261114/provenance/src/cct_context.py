"""CCT interaction bottleneck: RGB patches plus commanded/achieved transitions."""
import torch
from torch import nn


class InteractionEncoder(nn.Module):
    def __init__(self, visual_dim=1024, transition_dim=110, output_dim=4096,
                 width=512, queries=256, layers=6, heads=8, horizon=128, patches=196):
        super().__init__()
        self.horizon, self.patches = horizon, patches
        self.visual = nn.Linear(visual_dim, width)
        self.physical = nn.Sequential(nn.Linear(transition_dim, width), nn.GELU(), nn.Linear(width, width))
        self.temporal = nn.Embedding(horizon, width)
        self.spatial = nn.Embedding(patches, width)
        self.modality = nn.Embedding(2, width)
        self.trajectory = nn.Embedding(3, width)
        self.queries = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.input_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))

    def forward(self, visual, transitions, trajectory_ids):
        batch, times, patches, _ = visual.shape
        if times != self.horizon or patches != self.patches:
            raise ValueError("Unexpected context horizon or visual patch count")
        if transitions.shape[:2] != (batch, times) or trajectory_ids.shape != (batch, times):
            raise ValueError("Context streams are not aligned")
        if not ((trajectory_ids >= 0) & (trajectory_ids < 3)).all():
            raise ValueError("Expected three distinct trajectory slots")
        temporal = self.temporal(torch.arange(times, device=visual.device))[None]
        temporal = temporal + self.trajectory(trajectory_ids)
        spatial = self.spatial(torch.arange(patches, device=visual.device))[None, None]
        images = self.visual(visual) + temporal[:, :, None] + spatial + self.modality.weight[0]
        physical = self.physical(transitions) + temporal + self.modality.weight[1]
        inputs = self.input_norm(torch.cat((images.flatten(1, 2), physical), dim=1))
        latent = self.queries[None].expand(batch, -1, -1)
        latent = latent + self.cross_attention(self.query_norm(latent), inputs, inputs, need_weights=False)[0]
        for block in self.blocks:
            latent = block(latent)
        return self.output(latent)
