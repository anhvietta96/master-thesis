"""
This module contains the implementation of the VAE model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import TOKENS, input_dim, PAD_TOKEN, max_length, blosum62_tensor


def blosum62_loss(logits, targets, blosum62_tensor):
    """
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    pad_token = TOKENS.index(PAD_TOKEN)
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, len(TOKENS))             # (B*S, V)
    targets_flat = targets.view(-1)                       # (B*S)

    # Mask out padding positions
    mask = targets_flat != pad_token
    logits_masked = logits_flat[mask]                     # (N, V)
    targets_masked = targets_flat[mask].long()                   # (N,)

    # Convert logits to probabilities
    probs = F.softmax(logits_masked, dim=-1)              # (N, V)

    # Gather similarity vectors for each ground truth target
    blosum_rows = blosum62_tensor[targets_masked]           # (N, V)

    # Compute expected similarity
    expected_similarity = (probs * blosum_rows).sum(dim=-1)  # (N,)

    loss = 1.0 - expected_similarity.mean()               # Higher similarity → lower loss

    return loss

class VAE(nn.Module):
    def __init__(self, input_dim=164, latent_dim=128, hidden_dim=256):
        super(VAE, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.encoder_input = nn.Linear(input_dim, hidden_dim)
        self.encoder_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_hidden2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.decoder_output = nn.Linear(hidden_dim*2, input_dim)

    def encode(self, x):
        h = self.LeakyReLU(self.encoder_input(x))
        h = self.LeakyReLU(self.encoder_hidden(h))
        h = self.encoder_norm(h)
        mean = self.encoder_mean(h)
        logvar = self.encoder_logvar(h)
        logvar = torch.clamp(logvar, min=-15.0, max=15.0)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = self.LeakyReLU(self.decoder_hidden(z))
        h = self.LeakyReLU(self.decoder_hidden2(h))
        x_hat = self.decoder_output(h)
        return x_hat

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, mean, logvar

    def weight_init(self):
        def normal_init(m, mean, std):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                if m.bias.data is not None:
                    m.bias.data.zero_()

        for block in self._modules:
            try:
                for m in self._modules[block]:
                    normal_init(m,0,0.04)
            except:
                normal_init(block,0,0.04)

    @staticmethod
    def loss_fn(recon_x, x, mean, logvar, batch_size):
        recon_x = recon_x.view((batch_size, input_dim))
        x = x.view((batch_size, input_dim))
        recon_loss = F.cross_entropy(recon_x, x, reduction='mean') / max_length
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        #print(recon_loss,kl_div)
        return recon_loss, kl_div

class AE(nn.Module):
    def __init__(self, emb_dim=8, hidden_dim=256, latent_dim=128, seq_len=max_length, dropout=0.1):
        super(AE, self).__init__()
        self.vocab_size = len(TOKENS)
        self.seq_len = seq_len

        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=emb_dim, padding_idx=TOKENS.index(PAD_TOKEN)),
            nn.LayerNorm(emb_dim)
        )

        self.encoder = nn.Sequential(
            nn.Linear(seq_len * emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_dim,
                               kernel_size=5,
                               padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, seq_len,
                               kernel_size=3,
                               padding=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def encode(self, x):
        emb = self.embedding(x.int())  # (batch, seq_len, emb_dim)
        emb = emb.view(x.size(0), -1)  # flatten: (batch, seq_len * emb_dim)

        latent = self.encoder(emb)  # (batch, latent_dim)

        return latent

    def decode(self, latent):
        out = self.decoder(latent)  # (batch, seq_len * vocab_size)
        return out

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        out = out.view(x.size(0), self.seq_len, self.vocab_size)
        return out

    def weight_init(self):
        def normal_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

        for block in self._modules:
            try:
                for m in self._modules[block]:
                    normal_init(m)
            except:
                normal_init(block)

    @staticmethod
    def loss_fn(recon_logits, target, pad_token=0):
        return F.cross_entropy(
            recon_logits.view(-1, len(TOKENS)),  # (batch*seq_len, vocab_size)
            target.long().view(-1),                               # (batch*seq_len)
            ignore_index=TOKENS.index(PAD_TOKEN),
            reduction='mean'
        )

class CAE(nn.Module):
    def __init__(self, emb_dim=5, base_hidden_dim=64, latent_dim=128, seq_len=max_length):
        super(CAE, self).__init__()
        self.vocab_size = len(TOKENS)
        self.seq_len = seq_len
        self.base_hidden_dim = base_hidden_dim
        self.emb_dim = emb_dim

        self.embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=emb_dim,
                padding_idx=TOKENS.index(PAD_TOKEN)
            ),
            nn.LayerNorm(emb_dim)
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(emb_dim, base_hidden_dim, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(base_hidden_dim, base_hidden_dim*2, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(base_hidden_dim*2, base_hidden_dim*4, kernel_size=7, padding=3),
            nn.ReLU(),
        )

        self.encoder_fc = nn.Linear(base_hidden_dim*seq_len, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, base_hidden_dim*seq_len)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                base_hidden_dim*4,
                base_hidden_dim*2,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                base_hidden_dim*2,
                base_hidden_dim,
                kernel_size=7,
                padding=3,
                stride=2,
                output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                base_hidden_dim,
                emb_dim,
                kernel_size=7,
                padding=3,
                stride=2,
                output_padding=1
            ),
            nn.ReLU(),
        )

        self.output_proj = nn.Linear(emb_dim, len(TOKENS))

    def encode(self, x):
        emb = self.embedding(x.int())  # (batch, seq_len, emb_dim)
        emb = emb.permute(0, 2, 1)  # → (B, embed_dim, seq_len)
        z = self.encoder(emb)  # → (B, 256, 375)
        z = z.view(z.size(0), -1)  # → (B, 256*375)
        latent = self.encoder_fc(z)    # → (B, latent_dim)
        return latent

    def decode(self, latent):
        recon = self.decoder_fc(latent)       # → (B, 256*375)
        recon = recon.view(latent.size(0), self.base_hidden_dim * 4, self.seq_len//4)      # → (B, 256, 375)
        recon = self.decoder(recon)           # → (B, embed_dim, 3000)
        recon = recon.permute(0, 2, 1)         # → (B, 3000, embed_dim)
        logits = self.output_proj(recon)       # → (B, 3000, vocab_size)
        return logits

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        out = out.view(x.size(0), self.seq_len, self.vocab_size)
        return out

    def weight_init(self):
        def normal_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

        for block in self._modules:
            try:
                for m in self._modules[block]:
                    normal_init(m)
            except:
                normal_init(block)

    @staticmethod
    def loss_fn(recon_logits, target, blosum62_tensor):
        '''return F.cross_entropy(
            recon_logits.view(-1, len(TOKENS)),  # (batch*seq_len, vocab_size)
            target.long().view(-1),                               # (batch*seq_len)
            ignore_index=TOKENS.index(PAD_TOKEN),
            reduction='mean'
        )'''
        recon_loss = F.cross_entropy(
            recon_logits.view(-1, len(TOKENS)),  # (batch*seq_len, vocab_size)
            target.long().view(-1),                               # (batch*seq_len)
            ignore_index=TOKENS.index(PAD_TOKEN),
            reduction='mean'
        )
        return recon_loss, blosum62_loss(recon_logits, target, blosum62_tensor)


class CVAE(nn.Module):
    def __init__(self, base_hidden_dim=64, latent_dim=128, seq_len=max_length, kernel_size=7, padding=3):
        super(CVAE, self).__init__()
        emb_dim = blosum62_tensor.size(1)
        self.vocab_size = len(TOKENS)
        self.seq_len = seq_len
        self.base_hidden_dim = base_hidden_dim
        self.emb_dim = emb_dim

        '''self.embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=emb_dim,
                padding_idx=TOKENS.index(PAD_TOKEN)
            ),
            nn.LayerNorm(emb_dim)
        )'''

        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(blosum62_tensor, freeze=True),
            #nn.LayerNorm(emb_dim)
        )

        '''self.encoder = nn.Sequential(
            nn.Conv1d(emb_dim, base_hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(base_hidden_dim),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim, base_hidden_dim*2, kernel_size=kernel_size, padding=padding, stride=2),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim*2, base_hidden_dim*2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim*2, base_hidden_dim*4, kernel_size=kernel_size, padding=padding, stride=2),
            nn.BatchNorm1d(base_hidden_dim*4),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim*4, base_hidden_dim*4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(base_hidden_dim*4),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim*4, base_hidden_dim*8, kernel_size=kernel_size, padding=padding, stride=2),
            nn.BatchNorm1d(base_hidden_dim*8),
            nn.GELU(),
            nn.Flatten()
        )'''

        self.encoder = nn.Sequential(
            nn.Conv1d(emb_dim, base_hidden_dim*4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(base_hidden_dim*4),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim*4, base_hidden_dim*4, kernel_size=kernel_size, padding=padding, stride=2),
            nn.BatchNorm1d(base_hidden_dim*4),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim*4, base_hidden_dim*2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim*2, base_hidden_dim*2, kernel_size=kernel_size, padding=padding, stride=2),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.Conv1d(base_hidden_dim*2, base_hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(base_hidden_dim),
            nn.GELU()
        )

        self.encoder_mean = nn.Linear(base_hidden_dim*seq_len//4, latent_dim)
        self.encoder_logvar = nn.Linear(base_hidden_dim*seq_len//4, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, base_hidden_dim*seq_len//4)

        '''self.decoder = nn.Sequential(
            nn.Unflatten(1, (base_hidden_dim*4, seq_len//4)),
            nn.ConvTranspose1d(
                base_hidden_dim*8,
                base_hidden_dim*4,
                kernel_size=kernel_size,
                padding=padding,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm1d(base_hidden_dim*4),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*4,
                base_hidden_dim*4,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(base_hidden_dim*4),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*4,
                base_hidden_dim*2,
                kernel_size=kernel_size,
                padding=padding,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*2,
                base_hidden_dim*2,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*4,
                base_hidden_dim*2,
                kernel_size=kernel_size,
                padding=padding,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*2,
                base_hidden_dim*2,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*2,
                base_hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm1d(base_hidden_dim),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim,
                emb_dim,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
        )'''

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (base_hidden_dim, seq_len//4)),
            nn.ConvTranspose1d(
                base_hidden_dim,
                base_hidden_dim*2,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*2,
                base_hidden_dim*2,
                kernel_size=kernel_size,
                padding=padding,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm1d(base_hidden_dim*2),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*2,
                base_hidden_dim*4,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(base_hidden_dim*4),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*4,
                base_hidden_dim*4,
                kernel_size=kernel_size,
                padding=padding,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm1d(base_hidden_dim*4),
            nn.GELU(),
            nn.ConvTranspose1d(
                base_hidden_dim*4,
                emb_dim,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
        )

        self.output_proj = nn.Linear(emb_dim, len(TOKENS))

    def encode(self, x):
        emb = self.embedding(x.int())  # (batch, seq_len, emb_dim)
        if torch.isnan(emb).any():
            print("NaN after init embedding")
        emb = emb.permute(0, 2, 1)  # → (B, embed_dim, seq_len)
        z = self.encoder(emb)  # → (B, 256, 375)
        if torch.isnan(emb).any():
            print("NaN after encoder")
        z = z.view(z.size(0), -1)  # → (B, 256*375)
        mean = self.encoder_mean(z)    # → (B, latent_dim)
        if torch.isnan(mean).any():
            print("NaN after mean")
        logvar = self.encoder_logvar(z)    # → (B, latent_dim)
        if torch.isnan(logvar).any():
            print("NaN after logvar")
        '''if logvar.min().item() < -20 or logvar.max().item() > 20:
            print(f"Logvar overflow, min: {logvar.min().item()}, max: {logvar.max().item()}")'''
        logvar = torch.clamp(logvar, min=-20, max=20)
        return mean, logvar

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, latent):
        recon = self.decoder_fc(latent)       # → (B, 256*375)
        if torch.isnan(recon).any():
            print("NaN after decoder fc")
        #recon = recon.view(latent.size(0), self.base_hidden_dim * 4, self.seq_len//4)      # → (B, 256, 375)
        recon = self.decoder(recon)           # → (B, embed_dim, 3000)
        if torch.isnan(recon).any():
            print("NaN after decoder")
        recon = recon.permute(0, 2, 1)         # → (B, 3000, embed_dim)
        logits = self.output_proj(recon)       # → (B, 3000, vocab_size)
        return logits

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        if torch.isnan(z).any():
            print("NaN after reparametrize")
        out = self.decode(z)
        if torch.isnan(z).any():
            print("NaN after decoder")
        out = out.view(x.size(0), self.seq_len, self.vocab_size)
        return out, mean, logvar

    @staticmethod
    def loss_fn(recon_logits, target, mean, logvar):
        recon_loss = F.cross_entropy(
            recon_logits.view(-1, len(TOKENS)),  # (batch*seq_len, vocab_size)
            target.long().view(-1),                               # (batch*seq_len)
            ignore_index=TOKENS.index(PAD_TOKEN),
            reduction='mean'
        )
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        return recon_loss, kl_div
