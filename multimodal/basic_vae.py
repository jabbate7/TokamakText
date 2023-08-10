import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar


# Loss function
def vae_loss(decoded, x, mu, logvar):
    reconstruction_loss = nn.functional.mse_loss(decoded, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


def prepare_data():
    # Load and preprocess the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = YourCustomDataset(...)  # Replace with your own dataset loading logic
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train_vae(input_dim, epochs, lr):
    # Define the VAE model and optimizer
    input_dim = ...  # Replace with the dimensionality of your data
    vae = VAE()
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Training loop
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.view(-1, input_dim)  # Flatten the input if needed
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item() / len(data):.6f}')

        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(train_dataset):.6f}')

    print("Training finished!")


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    latent_dim = 20
    lr = 0.001
    epochs = 10

    vae = VAE(input_dim=10, latent_dim=512)
    dummy_data = torch.rand(3, 10)
    encoded = vae.encode(dummy_data)
    out = vae.forward(dummy_data)





