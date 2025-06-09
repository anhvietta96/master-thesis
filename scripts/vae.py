import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim):
        super(ProteinVAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        x_embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(x_embed)
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        # Repeat latent vector for each time step
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.decoder_rnn(z)
        return self.output_layer(output)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, x.size(1))
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # or MSE
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


vae = ProteinVAE(vocab_size=21, embed_dim=1000, hidden_dim=512, latent_dim=20)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in data_loader:
        batch = batch.to(device)
        recon_batch, mu, logvar = vae(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
