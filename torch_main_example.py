import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from disentanglement_lib.data.ground_truth import shapes3d,cars3d,mpi3d
from disentanglement_lib.evaluation.metrics import mig
# Custom Dataset Class for 3D Shapes
class CustomTensorDataset_3d(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform

    def __getitem__(self, index1):
        img1 = self.data_tensor[index1]
        if self.transform:
            img1 = self.transform(img1)
        return img1

    def __len__(self):
        return len(self.data_tensor)

# Load 3D Shapes Dataset
def load_3dshapes(): 
    cars_data = shapes3d.Shapes3D()
    data = cars_data.images
    data = torch.from_numpy(data).float().transpose(1, 3).transpose(2, 3)
    print("data",data.shape)
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset_3d
    train_data = dset(**train_kwargs)
    return train_data

# Latent class - represents the quantization layer
class Latent(nn.Module):
    def __init__(self, num_latents, num_values_per_latent, optimize_values=True):
        super().__init__()
        self.num_latents = num_latents
        self.num_values_per_latent = num_values_per_latent
        self.optimize_values = optimize_values
        self.values_per_latent = nn.ParameterList([
            nn.Parameter(torch.linspace(-0.5, 0.5, num_values)) 
            for num_values in num_values_per_latent
        ])

    def quantize(self, x, values):
        distances = torch.abs(x.unsqueeze(-1) - values)
        index = torch.argmin(distances, dim=-1)
        quantized = values[index]
        return quantized, index

    def forward(self, x):
        quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x, self.values_per_latent)]
        quantized = torch.stack([qi[0] for qi in quantized_and_indices])
        indices = torch.stack([qi[1] for qi in quantized_and_indices])
        return quantized, indices

# Encoder Class
class Encoder(nn.Module):
    def __init__(self, in_channels, out_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, out_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Decoder Class
class Decoder(nn.Module):
    def __init__(self, in_channels, out_shape):
        super().__init__()
        self.fc = nn.Linear(in_channels, 32 * 32 * 32)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 32, 32)
        print(f"After FC layer: {x.shape}")
        
        x = x.permute(0, 3, 1, 2)
        x = self.deconv1(x)
        print(f"After deconv1: {x.shape}")
        
        x = self.deconv2(x)
        print(f"After deconv2: {x.shape}")
        
        return x

# Autoencoder Class
class AE(nn.Module):
    def __init__(self, latent_partial, encoder_partial, decoder_partial, lambdas, x, key):
        super().__init__()
        self.latent = latent_partial(num_latents=10, num_values_per_latent=[10] * 10, optimize_values=True)
        self.encoder = encoder_partial(in_channels=x.shape[1], out_size=128)
        self.decoder = decoder_partial(in_channels=128, out_shape=x.shape)

    def forward(self, x):
        encoder_out = self.encoder(x)
        quantized, indices = self.latent(encoder_out)
        decoded = self.decoder(quantized)
        return decoded

# Training Loop for AE
def train_ae():
    # Dummy data for 3D Shapes, assume a batch size of 16 and image size 64x64
    train_data = load_3dshapes()
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model = AE(Latent, Encoder, Decoder, lambdas={'binary_cross_entropy': 1.0}, x=torch.randn(16, 3, 64, 64), key=None)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for data in train_loader:
            print("TRAIN data",data.shape)
            optimizer.zero_grad()
            output = model(data)
            print("output",output.shape)
            loss = F.mse_loss(output, data)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    train_ae()