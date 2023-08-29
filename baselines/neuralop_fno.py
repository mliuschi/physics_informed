# Author: Miguel Liu-Schiaffini (mliuschi@caltech.edu)
import torch
from torch import nn
from torch.nn import functional as F

# Official implementation of FNO
from neuralop.models import FNO3d

class FNO3D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,

        n_modes_width: int,
        n_modes_height: int,
        n_modes_depth: int,
        lifting_channels: int,
        projection_channels: int,
        n_layers: int,
        hidden_channels: int,
        use_mlp: bool,
        norm: bool,
        nonlinearity = F.gelu,
        mlp_dropout: float = 0,
        mlp_expansion: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 2 * modes because `neuralop` package defines modes differently
        self.FNO = FNO3d(
                n_modes_height = 2 * n_modes_height,
                n_modes_width = 2 * n_modes_width,
                n_modes_depth = 2 * n_modes_depth,
                hidden_channels = hidden_channels,
                in_channels = self.in_dim, 
                out_channels = self.out_dim,
                lifting_channels = lifting_channels,
                projection_channels = projection_channels,
                n_layers = n_layers,
                non_linearity = nonlinearity,
                use_mlp = use_mlp,
                mlp_dropout = mlp_dropout,
                mlp_expansion = mlp_expansion,
                norm = norm
            )

    def __repr__(self):
        return "FNO3D"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.FNO(x)

        return out