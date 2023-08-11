import os
import copy

import numpy as np
import tqdm
import h5py
from sklearn.model_selection import train_test_split
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),

        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim),
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


def train_vae(vae, input_dim, epochs, lr, train_loader, val_loader, test_loader):
    # Define the VAE model and optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    best_val_loss = float('inf')

    # Training loop
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        num_train = 0
        for batch_idx, data in enumerate(train_loader):
            num_train += data[0].shape[0]
            data = data[0].view(-1, input_dim).float()  # Flatten the input if needed
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            # if batch_idx % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item() / len(data):.6f}')

        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / num_train:.6f}')

        ####
        val_loss = 0
        num_val = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                num_val += data[0].shape[0]
                data = data[0].view(-1, input_dim).float()  # Flatten the input if needed
                recon_batch, mu, logvar = vae(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                val_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], VAL Loss: {val_loss / num_val:.6f}')
        ####

        ###
        if val_loss < best_val_loss:
            best_val_epoch = epoch
            best_val_loss = val_loss
            best_model = copy.deepcopy(vae.state_dict())

        print(best_val_epoch)
        early_stop = (500 < best_val_epoch) and best_val_epoch < (epoch + 50) and val_loss >= best_val_loss
        if early_stop:
            print("Early stopping)")
            break
        ###

        vae.load_state_dict(best_model)

    torch.save(vae.state_dict(), 'best_vae_model.pt')

    print("Training finished!")


def prep_data(
    fpath="/Users/youngsec/research/hack_fusion/example_194528.h5",
    signames=['t_ip_flat','ip_flat_duration','topology', 'poh', 'pech', 'pbeam', 'btor', 'btorsign', 'ip', 'ipsign', 'betanmax'],
    # signames=['t_ip_flat','ip_flat_duration','topology', 'poh', 'pbeam', 'btor', 'btorsign', 'ip', 'ipsign', 'betanmax'],
):

    topology_keys = ["TOP", "SNT", "SNB", "OUT", "MAR", "IN", "DN", "BOT"]
    data = h5py.File(fpath, 'r')
    shotnums = []
    values = []

    for k, v in tqdm.tqdm(data.items()):

        try:
            # deal with topology
            curr_v = [v[f"{sig}_sql"][()] for sig in signames]
            topology_idx = signames.index("topology")
            curr_topology = curr_v[topology_idx].decode("utf-8").strip()
            topology_int_key = topology_keys.index(curr_topology)
            curr_v[topology_idx] = float(topology_int_key)

            # deal with pech
            pech_idx = signames.index('pech')
            if not np.isfinite(curr_v[pech_idx]):
                curr_v[pech_idx] = 0

        except:
            continue

        if not all(np.isfinite(curr_v)):
            continue
            print(curr_v)

        # curr_v = [v[f"{sig}_sql"][()] for sig in signames]
        # curr_v = [float(v[f"{sig}_sql"][()]) for sig in signames]
        shotnums.append(k)
        values.append(curr_v)

    assert len(shotnums) == len(values)

    values = np.stack(values)

    np.save('shotnums_used.npy', np.array(shotnums))
    np.save('shot_data.npy', values)


def prepare_data(x_arr):
    num_data, num_features = x_arr.shape
    assert num_features == 11
    mu = np.mean(x_arr, axis=0)
    std = np.std(x_arr, axis=0)
    # Load and preprocess the dataset
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mu, std=std)
    # ])
    data = torch.from_numpy((x_arr - mu)/std)

    dataset = TensorDataset(data)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader


if __name__ == "__main__":
    # #####
    # # Hyperparameters
    # batch_size = 64
    # latent_dim = 5
    # lr = 0.0005
    # epochs = 5000
    # input_dim = 11

    # if os.path.exists('shotnums.npy'):
    #     shotnums = np.load('shotnums.npy')
    #     values = np.load('shot_data.npy')

    #     values_train = np.load("values_train.npy")
    #     values_val = np.load("values_val.npy")
    #     values_test = np.load("values_test.npy")
    #     shotnums_train = np.load("shotnums_train.npy")
    #     shotnums_val = np.load("shotnums_val.npy")
    #     shotnums_test = np.load("shotnums_test.npy")
    # else:
    #     prep_data()

    #     # Split into train and test
    #     values_train, values_test, shotnums_train, shotnums_test = train_test_split(
    #         values, shotnums, test_size=0.15, random_state=42)

    #     # Split train into train and validation
    #     values_train, values_val, shotnums_train, shotnums_val = train_test_split(
    #         values_train, shotnums_train, test_size=0.2, random_state=42)


    # train_loader = prepare_data(values_train)
    # val_loader = prepare_data(values_val)
    # test_loader = prepare_data(values_test)

    # vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    # breakpoint()
    # train_out = train_vae(vae, input_dim, epochs, lr, train_loader, val_loader, test_loader)
    # #####

    batch_size = 64
    latent_dim = 5
    lr = 0.0005
    epochs = 5000
    input_dim = 11
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    vae.load_state_dict(torch.load("best_vae_model.pt"))
    vae.eval()

    shotnums = np.load('shotnums.npy')
    values = np.load('shot_data.npy')
    values_train = np.load("values_train.npy")
    tr_mu = np.mean(values_train, axis=0)
    tr_std = np.std(values_train, axis=0)
    np.save('normalizer_mu.npy', tr_mu)
    np.save('normalizer_std.npy', tr_std)
    values = (values - tr_mu) / tr_std
    values_tensor = torch.from_numpy(values).float()
    # values_val = np.load("values_val.npy")
    # values_test = np.load("values_test.npy")
    # shotnums_train = np.load("shotnums_train.npy")
    # shotnums_val = np.load("shotnums_val.npy")
    # shotnums_test = np.load("shotnums_test.npy")

    with torch.no_grad():
        encodings = vae.encoder(values_tensor).numpy()
    encoding_dict = {}
    breakpoint()
    for i in range(encodings.shape[0]):

        encoding_dict[str(shotnums[i])] = encodings[i]

    # pkl.dump(encodings, open('vae_encodings.pkl', 'wb'))
    pkl.dump(encoding_dict, open('vae_encodings.pkl', 'wb'))


