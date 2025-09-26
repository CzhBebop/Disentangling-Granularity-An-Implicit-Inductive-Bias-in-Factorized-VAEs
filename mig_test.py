# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mig.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import shapes3d,cars3d,mpi3d
from disentanglement_lib.evaluation.metrics import mig
import numpy as np
import gin.tf
import model
import model_CONTRL
import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import gc
import lib.utils as utils
#from metric_helpers.loader import load_model_and_dataset
from metric_helpers.mi_metric import compute_metric_shapes, compute_metric_faces ,compute_metric_3dcars,compute_metric_3dshape,compute_metric_mpi

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

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def _identity_discretizer(target, num_bins):
  del num_bins
  return target

def load_3dcars():
    cars_data = cars3d.Cars3D()
    data = cars_data.images
    data = torch.from_numpy(data).float().transpose(1, 3).transpose(2, 3)
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset_3dcar
    train_data = dset(**train_kwargs)
    return train_data

def load_dsprite():
    root = '/disentanglement_lib-master/bin/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
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
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = DataLoader(dataset=train_data,
        batch_size=1000, shuffle=False, **kwargs)
    return train_loader

def load_model_and_dataset(root,net_type,data_type):
    if net_type =="contrl":
        if data_type =="dsprites":
            VAE_model = model_CONTRL.BetaVAE_H(z_dim=8, nc=1).cuda()
            VAE_model.load_state_dict(torch.load(root))
            dataset_train = load_dsprite()
        elif data_type =="3dcars":
            VAE_model = model_CONTRL.BetaVAE_H(z_dim=8, nc=3).cuda()
            VAE_model.load_state_dict(torch.load(root))
            dataset_train = load_3dcars()
        elif data_type =="3dshapes":
            VAE_model = model_CONTRL.BetaVAE_H(z_dim=8, nc=3).cuda()
            VAE_model.load_state_dict(torch.load(root))
            dataset_train = load_3dshapes()
        elif data_type =="3dmpi":
            VAE_model = model_CONTRL.BetaVAE_H(z_dim=8, nc=3).cuda()
            VAE_model.load_state_dict(torch.load(root))
            dataset_train = None
    elif net_type =="guide":
        if data_type =="dsprites":
            VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =1,raw_liner_dim = 4096,n_vae_dis = 8).cuda()
            VAE_model.load_state_dict(torch.load(root))
            dataset_train = load_dsprite()
        elif data_type =="3dcars":
            VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =3,raw_liner_dim = 12288,n_vae_dis = 8).cuda()
            VAE_model.load_state_dict(torch.load(root))
            dataset_train = load_3dcars()
        elif data_type =="3dshapes":
            VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =3,raw_liner_dim = 12288,n_vae_dis = 8).cuda()
            VAE_model.load_state_dict(torch.load(root))
            dataset_train = load_3dshapes()
        
    return VAE_model,dataset_train

def estimate_entropies(qz_samples, qz_params, q_dist,  args=None,n_samples=10000,weights=None,mutalInfo_total = None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """

    split_num = 1
    # Only take a sample subset of the samples
    if weights is None:
        qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())


    margin_entropies = torch.zeros(K).cuda()
    joint_entropies = torch.zeros(int(K/split_num)).cuda()
    margin_entropies_real = torch.zeros(int(K/split_num)).cuda()
    joint_entropy = torch.zeros(1).cuda()
    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size
        """print(logqz_i[:,0:4,:].shape)#logqz_i.shape (737280, 12, 10)
        input()"""
        if split_num>1:
            real_logqz_i = None
            for t in range(int(K/split_num)):
                if t == 0:
                    real_logqz_i = logqz_i[:,t:split_num,:].sum(1).unsqueeze(1)
                else:
                    real_logqz_i = torch.cat(
                                    (real_logqz_i,logqz_i[:,t*split_num:t*split_num+split_num,:].sum(1).unsqueeze(1)), 1)
            joint_entropies += - utils.logsumexp(real_logqz_i + weights, dim=0, keepdim=False).data.sum(1)#边缘联合熵
            """print(real_logqz_i.shape)
            input()"""
            #logqz_i = real_logqz_i
        logqz = logqz_i.sum(1)
        #joint_entropy += - utils.logsumexp(logqz + weights, dim=0, keepdim=False).data.sum(0)
        joint_entropy += - utils.logsumexp(logqz + weights, dim=0, keepdim=False).data.sum(0)
        margin_entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)#原始边缘熵
                    
        pbar.update(batch_size)
    pbar.close()
    print("margin_joint_entropies",joint_entropies)
    print("origin margin_entropies",margin_entropies)
    print("origin joint_entropy",joint_entropy)

    if split_num>1:
        for t in range(int(K/split_num)):
            if t == 0:
                margin_entropies_real = margin_entropies[t:split_num].sum().unsqueeze(0)
                print("margin_entropies_real")
            else:
                margin_entropies_real = torch.cat(
                                (margin_entropies_real,margin_entropies[t*split_num:t*split_num+split_num].sum().unsqueeze(0)), 0)
        print("I(p_xy/p_x p_y)",(margin_entropies_real - joint_entropies)/S)
        #input()
        #H(x) = E(-logp(x))
        I =  margin_entropies_real - joint_entropies #I(X,Y) = E[log[(p_xy)/p(x)p(y)]] = E[-(-log(p_xy))  +  ((-logp(x))  +  (-logp_y))]
        I_MASK = (I > 0).type(torch.uint8)   
        print("real I(p_xy/p_x p_y)",(I*I_MASK)/S)
        margin_entropies_real -= I*I_MASK #H_X + H_Y - I(X,Y) = H(X,Y)
        margin_entropies_real /= S #H_X + H_Y
        
    else:
        margin_entropies_real = margin_entropies/S
    
    ret_MASK = (margin_entropies_real > 0).type(torch.uint8)   
    margin_entropies_real = margin_entropies_real*ret_MASK
    print("ret entropies",margin_entropies_real)
    
    return margin_entropies_real


def mutual_info_metric_mpi(vae, dataset_loader, args=None):
    #dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=0, shuffle=False)
    split_num = 1
    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim  # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    #print('Computing q(z|x) distributions!!')
    qz_params = torch.Tensor(N, K, nparams)
    count = 0
    #print(count)
    n = 0
    mutalInfo_total = torch.zeros(int(K / split_num)).cuda()

    for xs in dataset_loader:
        #print(count)
        count += 1
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, 3, 64, 64).cuda(), volatile=True)
        qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        n += batch_size

    #4, 4, 2, 3, 3, 40, 40
    """\0 - floor color (10 different values)
      1 - wall color (10 different values)
      2 - object color (10 different values)
      3 - object size (8 different values)
      4 - object type (4 different values)
      5 - azimuth (15 different values)
    """
    qz_params = Variable(qz_params.view(4, 4, 2, 3, 3, 40, 40, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    #print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist, args=args)

    marginal_entropies = marginal_entropies.cpu()
    if split_num > 1:
        cond_entropies = torch.zeros(7, int(K / split_num))
    else:
        cond_entropies = torch.zeros(7, int(K))
        
    #print('Estimating conditional entropies for floor color.')
    for i in range(4):
        qz_samples_scale = qz_samples[i, :, :, :, :,:,:,:].contiguous()
        qz_params_scale = qz_params[i, :, :, :, :,:,:,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 4, K).transpose(0, 1),
            qz_params_scale.view(N // 4, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[0] += cond_entropies_i.cpu() / 4
        
    #print('Estimating conditional entropies for wall color.')
    for i in range(4):
        qz_samples_scale = qz_samples[:, i, :, :, :,:,:,:].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :,:,:,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 4, K).transpose(0, 1),
            qz_params_scale.view(N // 4, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[1] += cond_entropies_i.cpu() / 4
    
    #print('Estimating conditional entropies for object color.')
    for i in range(2):
        qz_samples_scale = qz_samples[:,:,i, :,:, :, :,:].contiguous()
        qz_params_scale = qz_params[:,:,i,:,:, :, :,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 2, K).transpose(0, 1),
            qz_params_scale.view(N // 2, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[2] += cond_entropies_i.cpu() / 2
        
    #print('Estimating conditional entropies for object size.')
    for i in range(3):
        qz_samples_scale = qz_samples[:,:,:, i,:, :, :,:].contiguous()
        qz_params_scale = qz_params[:,:,:,i,:, :, :,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 3 ,K).transpose(0, 1),
            qz_params_scale.view(N // 3, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[3] += cond_entropies_i.cpu() / 3
    
    #print('Estimating conditional entropies for object type.')
    for i in range(3):
        qz_samples_scale = qz_samples[:,:,:, :,i, :, :,:].contiguous()
        qz_params_scale = qz_params[:,:,:,:,i, :, :,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 3, K).transpose(0, 1),
            qz_params_scale.view(N // 3, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[4] += cond_entropies_i.cpu() / 3
        
    #print('Estimating conditional entropies for azimuth.')
    for i in range(40):
        qz_samples_scale = qz_samples[:,:,:, :,:, i, :,:].contiguous()
        qz_params_scale = qz_params[:,:,:,:,:, i, :,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40 ,K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[5] += cond_entropies_i.cpu() / 40
    

    for i in range(40):
        qz_samples_scale = qz_samples[:,:,:, :,:, :, i,:].contiguous()
        qz_params_scale = qz_params[:,:,:,:,:, :, i,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40 ,K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[6] += cond_entropies_i.cpu() / 40

    metric = compute_metric_mpi(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies




def mutual_info_metric_3dshape(vae, shapes_dataset, args= None):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=0, shuffle=False)
    split_num = 1
    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim  # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    #print('Computing q(z|x) distributions!!')
    qz_params = torch.Tensor(N, K, nparams)
    count = 0
    #print(count)
    n = 0
    mutalInfo_total = torch.zeros(int(K / split_num)).cuda()

    for xs in dataset_loader:
        #print(count)
        count += 1
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, 3, 64, 64).cuda(), volatile=True)
        #qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        mu,logvar = vae.encode(xs)
        qz_params[n:n + batch_size] = torch.cat((mu,logvar),-1).view(batch_size, vae.z_dim, nparams).data
        n += batch_size


    """\0 - floor color (10 different values)
      1 - wall color (10 different values)
      2 - object color (10 different values)
      3 - object size (8 different values)
      4 - object type (4 different values)
      5 - azimuth (15 different values)
    """
    qz_params = Variable(qz_params.view(10, 10, 10, 8, 4, 15, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    #print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist, args=args)

    marginal_entropies = marginal_entropies.cpu()
    if split_num > 1:
        cond_entropies = torch.zeros(6, int(K / split_num))
    else:
        cond_entropies = torch.zeros(6, int(K))
        
    #print('Estimating conditional entropies for floor color.')
    for i in range(10):
        qz_samples_scale = qz_samples[i, :, :, :, :,:,:].contiguous()
        qz_params_scale = qz_params[i, :, :, :, :,:,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 10, K).transpose(0, 1),
            qz_params_scale.view(N // 10, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[0] += cond_entropies_i.cpu() / 10
        
    #print('Estimating conditional entropies for wall color.')
    for i in range(10):
        qz_samples_scale = qz_samples[:, i, :, :, :,:,:].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :,:,:].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 10, K).transpose(0, 1),
            qz_params_scale.view(N // 10, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[1] += cond_entropies_i.cpu() / 10
    
    #print('Estimating conditional entropies for object color.')
    for i in range(10):
        qz_samples_scale = qz_samples[:,:,i, :,:, :, :].contiguous()
        qz_params_scale = qz_params[:,:,i,:,:, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 10, K).transpose(0, 1),
            qz_params_scale.view(N // 10, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[2] += cond_entropies_i.cpu() / 10
        
    #print('Estimating conditional entropies for object size.')
    for i in range(8):
        qz_samples_scale = qz_samples[:,:,:, i,:, :, :].contiguous()
        qz_params_scale = qz_params[:,:,:,i,:, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 8 ,K).transpose(0, 1),
            qz_params_scale.view(N // 8, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[3] += cond_entropies_i.cpu() / 8
    
    #print('Estimating conditional entropies for object type.')
    for i in range(4):
        qz_samples_scale = qz_samples[:,:,:, :,i, :, :].contiguous()
        qz_params_scale = qz_params[:,:,:,:,i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 4, K).transpose(0, 1),
            qz_params_scale.view(N // 4, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[4] += cond_entropies_i.cpu() / 4
        
    #print('Estimating conditional entropies for azimuth.')
    for i in range(15):
        qz_samples_scale = qz_samples[:,:,:, :,:, i, :].contiguous()
        qz_params_scale = qz_params[:,:,:,:,:, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 15 ,K).transpose(0, 1),
            qz_params_scale.view(N // 15, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[5] += cond_entropies_i.cpu() / 15



    metric = compute_metric_3dshape(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_shapes(vae, shapes_dataset,args= None):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1024, num_workers=0, shuffle=False)
    split_num = 1
    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions!!')
    qz_params = torch.Tensor(N, K, nparams)
    count = 0
    print(count)
    n = 0
    mutalInfo_total = torch.zeros(int(K/split_num)).cuda()
    
    for xs in dataset_loader:
        print(count)
        count+=1
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
        mu,logvar = vae.encode(xs)
        qz_params[n:n + batch_size] = torch.cat((mu,logvar),-1).view(batch_size, vae.z_dim, nparams).data
        """qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data"""
        n += batch_size

    qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist,args=args)

    marginal_entropies = marginal_entropies.cpu()
    if split_num>1:
        cond_entropies = torch.zeros(4, int(K/split_num))
    else:
        cond_entropies = torch.zeros(4, int(K))
    print('Estimating conditional entropies for scale.')
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 6, K).transpose(0, 1),
            qz_params_scale.view(N // 6, K, nparams),
            vae.q_dist,args=args)

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40, K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams),
            vae.q_dist,args=args)

        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            vae.q_dist,args=args)

        cond_entropies[2] += cond_entropies_i.cpu() / 32

    print('Estimating conditional entropies for pox y.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            vae.q_dist,args=args)

        cond_entropies[3] += cond_entropies_i.cpu() / 32


    metric = compute_metric_shapes(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_3dcars(vae, shapes_dataset, args=None):
    dataset_loader = DataLoader(shapes_dataset, batch_size=64, num_workers=0, shuffle=False)
    split_num = 1
    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim  # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions!!')
    qz_params = torch.Tensor(N, K, nparams)
    count = 0
    print(count)
    n = 0
    mutalInfo_total = torch.zeros(int(K / split_num)).cuda()

    for xs in dataset_loader:
        print(count)
        count += 1
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, 3, 64, 64).cuda(), volatile=True)
        mu,logvar = vae.encode(xs)
        qz_params[n:n + batch_size] = torch.cat((mu,logvar),-1).view(batch_size, vae.z_dim, nparams).data
        #qz_params[n:n + batch_size] = vae.encode(xs).view(batch_size, vae.z_dim, nparams).data
        n += batch_size



    qz_params = Variable(qz_params.view(4, 24, 183, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist, args=args)

    marginal_entropies = marginal_entropies.cpu()
    if split_num > 1:
        cond_entropies = torch.zeros(3, int(K / split_num))
    else:
        cond_entropies = torch.zeros(3, int(K))
    print('Estimating conditional entropies for elevation.')
    for i in range(4):
        qz_samples_scale = qz_samples[i, :, :, :].contiguous()
        qz_params_scale = qz_params[i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 4, K).transpose(0, 1),
            qz_params_scale.view(N // 4, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for azimuth.')
    for i in range(24):
        qz_samples_scale = qz_samples[:, i, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 24, K).transpose(0, 1),
            qz_params_scale.view(N // 24, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for type.')
    for i in range(183):
        qz_samples_scale = qz_samples[:, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 183, K).transpose(0, 1),
            qz_params_scale.view(N // 183, K, nparams),
            vae.q_dist, args=args)

        cond_entropies[2] += cond_entropies_i.cpu() / 183

    metric = compute_metric_3dcars(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies

import json
class MIGTest(absltest.TestCase):

  def test_metric_contrl(self):
    net_type = 'contrl'#"contrl" #"guide"
    
    baseRoot = "/contrlVAE_"
    total_scores_3dcars = []
    total_scores_3dshapes = []
    total_scores_dsprites = []

    """for i in range(1):
        vae, dataset = load_model_and_dataset(baseRoot+"3dcars_"+str(i)+".pth",net_type,"3dcars")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_3dcars(vae, dataset)
        mig= metric.item()
        print("mig",mig )
        total_scores_3dcars.append(mig)
    with open(baseRoot+"3dcars_MIG_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dcars, f)"""
    


    mpi_dataset = load_3dmpi()
    for i in range(100):
        vae, dataset = load_model_and_dataset(baseRoot+"3dmpi_"+str(i)+".pth",net_type,"3dmpi")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_mpi(vae, mpi_dataset)
        mig= metric.item()
        print("mig",mig)
        total_scores_3dshapes.append(mig)
        vae=None
        dataset = None
        torch.cuda.empty_cache()
        gc.collect()
    with open(baseRoot+"3dmpi_MIG_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dshapes, f)

    """for i in range(1):
        vae, dataset = load_model_and_dataset(baseRoot+"dsprites_"+str(i)+".pth",net_type,"dsprites")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_shapes(vae, dataset)
        mig= metric.item()
        print("mig",mig)
        total_scores_dsprites.append(mig)
    with open(baseRoot+"dsprites_MIG_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_dsprites, f)"""


    """for i in range(1):
        vae, dataset = load_model_and_dataset(baseRoot+"3dcars_"+str(i)+".pth",net_type,"3dcars")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_3dcars(vae, dataset)
        mig= metric.item()
        print("mig",mig )
        total_scores_3dcars.append(mig)
    with open(baseRoot+"3dcars_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dcars, f)
    

    for i in range(1):
        vae, dataset = load_model_and_dataset(baseRoot+"3dshapes_"+str(i)+".pth",net_type,"3dshapes")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_3dshape(vae, dataset)
        mig= metric.item()
        print("mig",mig)
        total_scores_3dshapes.append(mig)
    with open(baseRoot+"3dshapes_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dshapes, f)

    for i in range(1):
        vae, dataset = load_model_and_dataset(baseRoot+"dsprites_"+str(i)+".pth",net_type,"dsprites")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_shapes(vae, dataset)
        mig= metric.item()
        print("mig",mig)
        total_scores_dsprites.append(mig)
    with open(baseRoot+"dsprites_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_dsprites, f)
    """
    
  """def test_metric_guide(self):
    net_type = 'guide'#"contrl" #"guide"
    baseRoot = "/factor_Guided/unGuidedVAE_"
    #baseRoot = "/ContrlVAE/contrlVAE_"
    total_scores_3dcars = []
    total_scores_3dshapes = []
    for i in range(100):
        vae, dataset = load_model_and_dataset(baseRoot+"3dshapes_"+str(i)+".pth",net_type,"3dshapes")
        metric, marginal_entropies, cond_entropies = mutual_info_metric_3dshape(vae, dataset)
        mig= metric.item()
        print("mig",mig)
        total_scores_3dshapes.append(mig)
        vae=None
        dataset = None
        torch.cuda.empty_cache()
        gc.collect()
    with open(baseRoot+"3dshapes_MIG_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dshapes, f)"""




def mig_call():
    absltest.main()

if __name__ == "__main__":
  mig_call()
