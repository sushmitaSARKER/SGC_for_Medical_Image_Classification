import torch
import torch.nn as nn
from models.unet_parts import *
from models.embedding import *

class UNetScoreModel_conditional(nn.Module):
    """
    A conditional UNet-based score model for score-based generative modeling.

    Args:
        marginal_prob_std: Function to compute the marginal probability standard deviation.
        n_channels (int): Number of input channels. Default is 3.
        n_classes (int): Number of classes for conditional embedding. Default is 2.
        bilinear (bool): Whether to use bilinear interpolation in upsampling. Default is False.
        channels (list): List of channel sizes for each layer. Default is [32, 64, 128, 256, 512].
        embed_dim (int): Dimension of the time and label embeddings. Default is 256.
        sde: Optional SDE model for scaling the output. Default is None.
    """

    def __init__(
        self,
        marginal_prob_std,
        n_channels=3,
        n_classes=2,
        bilinear=False,
        channels=[32, 64, 128, 256, 512],
        embed_dim=256,
        sde=None,
    ):
        super(UNetScoreModel_conditional, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.embed_dim = embed_dim
        self.marginal_prob_std = marginal_prob_std
        self.sde = sde

        # Time embedding layer
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Label embedding layer
        if n_classes is not None:
            self.label_emb = nn.Embedding(n_classes, embed_dim)

        # Encoder
        self.conv1 = DoubleConv(n_channels, channels[0])
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.down2 = Down(channels[0], channels[1])
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.down3 = Down(channels[1], channels[2])
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.down4 = Down(channels[2], channels[3])
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        factor = 2 if bilinear else 1
        self.down5 = Down(channels[3], channels[4] // factor)
        self.dense5 = Dense(embed_dim, channels[4] // factor)
        self.gnorm5 = nn.GroupNorm(32, num_channels=channels[4] // factor)

        # Decoder
        self.up4 = Up(channels[4], channels[3] // factor, bilinear)
        self.t_dense4 = Dense(embed_dim, channels[3] // factor)
        self.t_gnorm4 = nn.GroupNorm(32, num_channels=channels[3] // factor)

        self.up3 = Up(channels[3], channels[2] // factor, bilinear)
        self.t_dense3 = Dense(embed_dim, channels[2] // factor)
        self.t_gnorm3 = nn.GroupNorm(32, num_channels=channels[2] // factor)

        self.up2 = Up(channels[2], channels[1] // factor, bilinear)
        self.t_dense2 = Dense(embed_dim, channels[1] // factor)
        self.t_gnorm2 = nn.GroupNorm(32, num_channels=channels[1] // factor)

        self.up1 = Up(channels[1], channels[0] // factor, bilinear)
        self.t_dense1 = Dense(embed_dim, channels[0] // factor)
        self.t_gnorm1 = nn.GroupNorm(32, num_channels=channels[0] // factor)

        # Output layer
        self.outc = OutConv(channels[0], n_channels)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t, y=None):
        """
        Forward pass of the conditional UNet score model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width).
            t (torch.Tensor): Time tensor of shape (batch_size,).
            y (torch.Tensor, optional): Class labels of shape (batch_size,). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_channels, height, width).
        """
        # Time embedding
        embed_wo_label = self.act(self.embed(t))

        # Add label embedding if provided
        if y is not None:
            embed = embed_wo_label + self.label_emb(y)
        else:
            embed = embed_wo_label

        # Encoder
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.down2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.down3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.down4(h3) + self.dense4(embed)))
        h5 = self.act(self.gnorm5(self.down5(h4) + self.dense5(embed)))

        # Decoder
        h = self.act(self.t_gnorm4(self.up4(h5, h4) + self.t_dense4(embed)))
        h = self.act(self.t_gnorm3(self.up3(h, h3) + self.t_dense3(embed)))
        h = self.act(self.t_gnorm2(self.up2(h, h2) + self.t_dense2(embed)))
        h = self.act(self.t_gnorm1(self.up1(h, h1) + self.t_dense1(embed)))

        # Output
        h = self.outc(h)

        # Scale output based on marginal probability standard deviation
        if self.sde is not None:
            h = h / self.marginal_prob_std(x, t)[1][:, None, None, None]
        else:
            h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h