# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from vqvae import VQVAE
from disentanglement_lib.data.ground_truth import shapes3d,cars3d,mpi3d

torch.set_printoptions(linewidth=160)
class CustomTensorDataset_3d(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        img1 = self.data_tensor[index1]
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1

    def __len__(self):
        return self.data_tensor.size(0)
class CustomTensorDataset_3dcar(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        img1 = self.data_tensor[index1]
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1

    def __len__(self):
        return self.data_tensor.size(0)

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

import os
def load_dsprite():
    root = 'E:/python3.8.10/disentanglement_lib-master/bin/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    if not os.path.exists(root):
        import subprocess
        print('Now download dsprites-dataset')
        subprocess.call(['./download_dsprites.sh'])
        print('Finished')
    data = np.load(root, encoding='bytes')
    data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset
    train_data = dset(**train_kwargs)
    return train_data


def load_3dcars():
    cars_data = cars3d.Cars3D()
    data = cars_data.images
    data = torch.from_numpy(data).float().transpose(1, 3).transpose(2, 3)
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset_3dcar
    train_data = dset(**train_kwargs)
    return train_data

def load_3dshapes():
    cars_data = shapes3d.Shapes3D()
    data = cars_data.images
    data = torch.from_numpy(data).float().transpose(1, 3).transpose(2, 3)
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset_3d
    train_data = dset(**train_kwargs)
    return train_data


def load_3dmpi():
    cars_data = mpi3d.MPI3D()
    data = cars_data.images
    data = torch.from_numpy(data).float().transpose(1, 3).transpose(2, 3)
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset_3d
    train_data = dset(**train_kwargs)
    
    return train_data

def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")
DATANAME = "dsprite"

def main(COUNT = 0,DATANAME='CAR',EPOCH=100):
    if DATANAME == "dsprite":
        in_channels = 1
    else:
        in_channels = 3
    # Initialize model.
    device = torch.device("cuda:0")
    use_ema = True
    model_args = {
        "in_channels": in_channels,
        "num_hiddens": 128,
        "num_downsampling_layers": 4,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": 16,
        "num_embeddings": 64,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = VQVAE(**model_args).to(device)

    # Initialize dataset.
    batch_size = 256
    workers = 0
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    data_root = "../data"
    #train_dataset = CIFAR10(data_root, True, transform, download=True)
    #train_dataset = load_3dshapes()
    if DATANAME == "CAR":
        train_dataset = load_3dcars()
    elif DATANAME == "SHAPE":
        train_dataset = load_3dshapes()
    elif DATANAME == "dsprite":
        train_dataset = load_dsprite()
    elif DATANAME == "3dmpi":
        train_dataset = load_3dmpi()

    
    #print("dataset_train",train_dataset)
    #train_data_variance = np.var(dataset_train / 255)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )

    # Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation
    # Learning".
    beta = 0.25

    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    lr = 3e-4
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()

    # Train model.
    epochs = EPOCH
    eval_every = 10
    best_train_loss = float("inf")
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(train_loader):
            optimizer.zero_grad()
            #imgs = train_tensors[0].to(device)
            imgs = train_tensors.to(device)
            #print("imgs",imgs.shape)
            out = model(imgs)
            recon_error = criterion(out["x_recon"], imgs)
            total_recon_error += recon_error.item()
            loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                loss += out["dictionary_loss"]
            #print("encoding_indices",out["encoding_indices"])
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

            if ((batch_idx + 1) % eval_every) == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                total_train_loss /= n_train
                if total_train_loss < best_train_loss:
                    best_train_loss = total_train_loss

                print(f"total_train_loss: {total_train_loss}")
                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {total_recon_error / n_train}\n")

                total_train_loss = 0
                total_recon_error = 0
                n_train = 0
    torch.save(model.state_dict(), "E:\\python3.8.10\\disentanglement_metric\\vqvae_model\\best_model_"+DATANAME+"_"+str(COUNT)+".pth")
    # Generate and save reconstructions.
    """model.eval()

    valid_dataset = CIFAR10(data_root, False, transform, download=True)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )

    with torch.no_grad():
        for valid_tensors in valid_loader:
            break

        save_img_tensors_as_grid(valid_tensors[0], 4, "true")
        save_img_tensors_as_grid(model(valid_tensors[0].to(device))["x_recon"], 4, "recon")"""
for DATASET in ["SHAPE"]:
    for i in range(6):
        if DATASET =="CAR":
            EPOCH = 500
        elif DATASET =="SHAPE" or DATASET =="3dmpi":
            EPOCH = 100
        elif DATASET =="dsprite":
            EPOCH = 50
        main(COUNT = i,DATANAME=DATASET,EPOCH=EPOCH)