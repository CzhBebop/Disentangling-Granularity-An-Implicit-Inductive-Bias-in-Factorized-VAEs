import os
import time
import math
import json
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow

from disentanglement_metrics import mutual_info_metric_3dshape
import torch.nn.functional as F
from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401
import time
import seaborn as sns
import demjson
import matplotlib

from dataset_3dshape import return_traindata
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
sns.set()
#width = 28
width = 64
#fc_width = 784
fc_width = 4096
class MLPEncoder(nn.Module):
    def __init__(self, output_dim,nur_num,layer):
        super(MLPEncoder, self).__init__()

        # Build Encoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(3, out_channels=16,
                          kernel_size= 4, stride= 2, padding  = 1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),

                nn.Conv2d(16, out_channels=32,
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),

                nn.Conv2d(32, out_channels=64,
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU()))


        self.cov_encoder = nn.Sequential(*modules)
        ##########################################################

        self.layer = layer
        self.layer_list = []

        self.fc_1 = nn.Linear(nur_num, nur_num)
        self.layer_list.append(self.fc_1)

        self.fc_2 = nn.Linear(nur_num, nur_num)
        self.layer_list.append(self.fc_2)

        """self.fc_4 = nn.Linear(nur_num, nur_num)
        self.layer_list.append(self.fc_4)

        self.fc_5 = nn.Linear(nur_num, nur_num)
        self.layer_list.append(self.fc_5)

        self.fc_6 = nn.Linear(nur_num, nur_num)
        self.layer_list.append(self.fc_6)

        self.fc_7 = nn.Linear(nur_num, nur_num)
        self.layer_list.append(self.fc_7)

        self.fc_8 = nn.Linear(nur_num, nur_num)
        self.layer_list.append(self.fc_8)

        self.fc_9 = nn.Linear(nur_num, nur_num)
        self.layer_list.append(self.fc_9)"""

        self.output_dim = output_dim

        self.fc1 = nn.Linear(fc_width,nur_num)
        #self.fc2 = nn.Linear(nur_num, nur_num)
        self.fc3 = nn.Linear(nur_num, output_dim)
        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        result = self.cov_encoder(x)
        h = result.view(-1,fc_width)
        h = self.act(self.fc1(h))
        for i in range(self.layer):
            h = self.act(self.layer_list[i](h))
        #h = self.act(self.fc2(h))
        h = self.fc3(h)

        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim,nur_num,layer):
        super(MLPDecoder, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(64,
                                   32,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   ),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),

                nn.ConvTranspose2d(32,
                                   16,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   ),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),

                nn.ConvTranspose2d(16,
                                   3,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   )))

        self.cov_decoder = nn.Sequential(*modules)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, nur_num),
            nn.Tanh(),
            nn.Linear(nur_num, nur_num),
            nn.Tanh(),
            nn.Linear(nur_num, nur_num),
            nn.Tanh(),
            nn.Linear(nur_num, fc_width),
            nn.Tanh(),
        )


    def forward(self, z):
        h = z.view(z.size(0),-1)
        h = self.net(h)
        cov_in = h.view(z.size(0), 64, 8, 8)
        mu_img = self.cov_decoder(cov_in)
        return mu_img

class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 3, width, width)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        #self.conv_z(h).view(x.size(0), -1)
        z = self.conv_z(h).view(x.size(0), -1)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 3, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class VAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, conv=False, mss=False,newTC = False,contraintInfo = False,nur_num = 1000,layer=1):
        super(VAE, self).__init__()
        self.newTC = newTC
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.contraintInfo = contraintInfo
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams,nur_num,layer)
            self.decoder = MLPDecoder(z_dim,nur_num,layer)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 3, width, width)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params 

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 3, width, width)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params 

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def function_z(self,x):
        x = x.view(x.size(0), 3, width, width)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size,split_num,beta):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        x = x.view(batch_size, 3, width, width)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)#decode(x_recon, x_params) encode(zs, z_params)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        #print("logpx",logpx.shape,"logpz",logpz.shape,"logqz_condx",logqz_condx.shape)
        elbo = logpx + logpz - logqz_condx 

        """if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()"""
        #print("zs shape", zs.shape,"z_params",z_params.shape)
        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))

        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        ) #q(z(ni)|nj)   z(ni) is a sample from q(z|ni)
        #temp = _logqz.select(-1, 0)
        #print(temp,_logqz)
        log_mu = None
        important_sample_mu = None
        Fake_important_sample_mu = None
        log_double_loc_prodmarginals = None
        constraint_mu = None#
        log_double_loc_list = []
        log_loc_qz_list = []
        #print("self.z_dim",self.z_dim,type(self.z_dim))
        if self.newTC:
            heatMap = np.zeros((int(self.z_dim/split_num) +1 ,int(self.z_dim/split_num) +1 ))
        elif self.tcvae:
            heatMap = np.zeros((int(self.z_dim), int(self.z_dim)))
        if self.newTC:#PMVAE
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            if self.z_dim % split_num==0:#
                ranger = int(self.z_dim/split_num)
            else:#
                ranger = int(self.z_dim/split_num) +1
            for i in range(ranger):
                log_loc_qz = None
                if (i+1)*split_num>self.z_dim:
                    ranger_sec = self.z_dim - (i)*split_num
                    #print(self.z_dim)
                else:
                    ranger_sec = split_num

                #print(split_num)
                for j in range(ranger_sec):
                    logq_zi = _logqz.select(-1, i*split_num+j).view(batch_size, batch_size, 1)
                    if isinstance(log_loc_qz,type(None)):
                        log_loc_qz = logq_zi
                    else:
                        log_loc_qz = torch.cat((log_loc_qz, logq_zi), -1)
                    
                logqz_loc_prodmarginals = (logsumexp(log_loc_qz, dim=1, keepdim=False) - math.log(
                    batch_size * dataset_size)).sum(1)
                logqz_loc = (logsumexp(log_loc_qz.sum(2), dim=1, keepdim=False) - math.log(
                    batch_size * dataset_size))
                # print("logqz_loc",logqz_loc)
                if log_double_loc_prodmarginals == None:
                    log_double_loc_prodmarginals = logqz_loc
                else:
                    log_double_loc_prodmarginals = log_double_loc_prodmarginals + logqz_loc
                """if log_mu == None:
                    log_mu = torch.exp(logqz_loc) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)
                else:
                    log_mu = torch.cat(
                        (log_mu, torch.exp(logqz_loc) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)), 0)"""
                if important_sample_mu == None:
                    important_sample_mu =  (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)
                else:
                    important_sample_mu = torch.cat(
                        (important_sample_mu,  (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)), 0)
                if Fake_important_sample_mu == None:
                    Fake_important_sample_mu = torch.exp(logqz_loc) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)
                else:
                    Fake_important_sample_mu = torch.cat(
                        (Fake_important_sample_mu, torch.exp(logqz_loc) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)), 0)
                log_double_loc_list.append(logqz_loc)
                log_loc_qz_list.append(log_loc_qz)
                #heatMap[j][i] = (torch.exp(logqz_loc) * (logqz_loc - logqz_loc_prodmarginals)).detach().sum()
            for len_i in range(len(log_double_loc_list)-1):
                for len_j in range(len(log_double_loc_list)):
                    if len_j>len_i:
                        log_qua_loc_qz = torch.cat((log_loc_qz_list[len_i],log_loc_qz_list[len_j]), -1)
                        logqz_qua_loc = (logsumexp(log_qua_loc_qz.sum(2), dim=1, keepdim=False) - math.log(
                            batch_size * dataset_size))
                        heatMap[len_j][len_i] = (torch.exp(logqz) * (logqz_qua_loc - (log_double_loc_list[len_i]+log_double_loc_list[len_j]))).detach().sum()
                        if constraint_mu == None:
                            constraint_mu = torch.exp(logqz) * (logqz_qua_loc - (log_double_loc_list[len_i]+log_double_loc_list[len_j])).unsqueeze(0)
                        else:
                            constraint_mu = torch.cat(
                                (constraint_mu, torch.exp(logqz) * (logqz_qua_loc - (log_double_loc_list[len_i]+log_double_loc_list[len_j])).unsqueeze(0)), 0)
        elif self.tcvae:#TCVAE

            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))

            for i in range(self.z_dim):
                logq_zi = _logqz.select(-1, i).view(batch_size, batch_size, 1)
                for j in range(self.z_dim):
                    if j > i:
                        logq_zj = _logqz.select(-1, j).view(batch_size, batch_size, 1)
                        log_loc_qz = torch.cat((logq_zi, logq_zj), -1)
                        logqz_loc_prodmarginals = (logsumexp(log_loc_qz, dim=1, keepdim=False) - math.log(
                            batch_size * dataset_size)).sum(1)
                        logqz_loc = (logsumexp(log_loc_qz.sum(2), dim=1, keepdim=False) - math.log(
                            batch_size * dataset_size))
                        #print("logqz_loc",logqz_loc)
                        if i%2==0 and j == i+1:
                            if log_double_loc_prodmarginals == None:
                                log_double_loc_prodmarginals = logqz_loc
                            else:
                                log_double_loc_prodmarginals = log_double_loc_prodmarginals + logqz_loc
                            """if log_mu == None:
                                log_mu = torch.exp(logqz_loc) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)
                            else:
                                log_mu = torch.cat(
                                    (log_mu, torch.exp(logqz_loc) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)), 0)"""
                            if important_sample_mu == None:
                                important_sample_mu = torch.exp(logqz) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)
                            else:
                                important_sample_mu = torch.cat(
                                    (important_sample_mu, torch.exp(logqz) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)), 0)
                        heatMap[j][i] = (torch.exp(logqz) * (logqz_loc - logqz_loc_prodmarginals)).detach().sum()
                        if constraint_mu == None:
                            constraint_mu = torch.exp(logqz) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)
                        else:
                            constraint_mu = torch.cat(
                                (constraint_mu, torch.exp(logqz) * (logqz_loc - logqz_loc_prodmarginals).unsqueeze(0)), 0)
            #print("important_sample_mu",important_sample_mu)
        #print("logz shape",_logqz.shape)
        if not self.mss: 
            # minibatch weighted sampling

            #print("minibatch weighted sampling")
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            #log_mu = log_mu.sum(0)
        elif self.newTC:
            #print("here")
            """print("newTC weighted sampling")
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))"""

            #print("newTC weighting sampling")
            #print("important_sample_mu shape",important_sample_mu.shape)
            
            important_sample_mu_ret =  important_sample_mu.detach()
            constraint_mu = constraint_mu.mean(0)
            important_sample_mu = important_sample_mu.mean(-1)
            log_double_loc_ = (logqz - log_double_loc_prodmarginals)
            #log_mu = log_mu.sum(0)

            #input()"
        else:#
            #print("TCVAE weighting sampling")
            #print(constraint_mu.shape)

            constraint_mu = constraint_mu.mean(0)
            important_sample_mu = important_sample_mu.sum(0)
            log_double_loc_ = torch.exp(logqz) * (logqz - log_double_loc_prodmarginals)
            #log_mu = log_mu.sum(0)


        if self.newTC:
            #print("new VAE")
            if self.contraintInfo==1:
                #print("log_double_loc_prodmarginals",log_double_loc_prodmarginals)
                #print("jtc")
                modified_elbo = 1*logpx - \
                    1*(logqz_condx - logqz)  - \
                    beta *  (log_double_loc_)-\
                    1*(logqz_prodmarginals - logpz)
            #print("important_sample_mu", important_sample_mu.sum().detach())
            elif self.contraintInfo==2:
                #print("jtc+mu")
                modified_elbo = 1 * logpx - \
                                1 * (logqz_condx - logqz) - \
                                beta * (log_double_loc_) +\
                                beta/2 * important_sample_mu.sum() -\
                                1 * (logqz_prodmarginals - logpz)
            elif self.contraintInfo==3:
                #print("tc")
                modified_elbo = 1 * logpx - \
                                (logqz_condx - logqz)- \
                                beta *(logqz - logqz_prodmarginals) - \
                                (logqz_prodmarginals - logpz)
            elif self.contraintInfo==4:
                #print("beta-vae")
                modified_elbo = logpx - beta * (
                        (logqz_condx - logpz) -
                        (logqz_prodmarginals - logpz)
                )


            fake_TC =(Fake_important_sample_mu.sum() + (torch.exp(logqz) * log_double_loc_).sum()).detach()
            real_TC = (important_sample_mu.sum() + (log_double_loc_).mean()).detach()
            MU = (logqz_condx - logqz).detach()
            TC = (logqz - logqz_prodmarginals).mean().detach()
            ADD = (log_double_loc_).mean().detach()
            REC = logpx.detach()
        elif not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                        (logqz_condx - logpz) -
                        self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                        (logqz - logqz_prodmarginals) +
                        (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
            MU = torch.exp(logqz)*(logqz - logqz_prodmarginals)
        else:
            #print("TC VAE")
            if self.include_mutinfo:
                modified_elbo = 1 * logpx.mean()- \
                    1 * (logqz_condx - logqz).mean() - \
                    10 * (important_sample_mu.sum() + log_double_loc_.sum()) -\
                    1 * (logqz_prodmarginals - logpz).mean()

            MU = (logqz_condx - logqz).detach()
            TC = (torch.exp(logqz)*(logqz - logqz_prodmarginals)).detach()
            ADD = log_double_loc_.detach()
            REC = (logpx).detach()
        return modified_elbo, elbo.detach(),MU,TC,real_TC,fake_TC,ADD,REC,heatMap,important_sample_mu_ret


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    elif args.dataset == 'minest':
        train_set = datasets.MNIST(root="/data/",
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
        """train_dataset = datasets.CelebA(root='/data/',
                                        split='train',
                                        transform=transforms.ToTensor(),
                                        download=True)"""
    elif args.dataset == "cars3d":
        train_set = return_traindata()
    elif args.dataset == "shape3d":
        train_set = return_traindata()
        
        
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))
    #print("train_set",train_set)
    #input()
    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader



win_samples = None
win_test_reco = None
win_latent_walk = None
win_joint_latend_walk_1 = None
win_joint_latend_walk_2 = None
win_train_elbo = None
win_TC = None
win_MU = None
win_add = None
win_REC = None
def display_samples(model, x, vis,gap):
    global  win_latent_walk,win_joint_latend_walk_1,win_joint_latend_walk_2,win_test_reco

    """
    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    images = torch.tensor([item.cpu().detach().numpy() for item in sample_mu.view(-1, 1, 64, 64)]).cuda()
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'},win = win_samples)
    """
    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([
        test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
    win_test_reco = vis.images(
        torch.tensor([item.cpu().detach().numpy() for item in test_reco_imgs.contiguous().view(-1, 1, 64, 64)]).cuda()
        , 10, 2,
        opts={'caption': 'test reconstruction image'}, win=win_test_reco)
    
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    #print("z_dim",z_dim)
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = torch.tensor([item.cpu().detach().numpy() for item in torch.cat(xs, 0)]).cuda()

    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)
    #joint latend walks patterm1

    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(0,z_dim,gap):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i:i+gap] = 1
        #vec[:, i+1] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i:i+gap] = 0
        #zs_delta[:, :, i+1] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = torch.tensor([item.cpu().detach().numpy() for item in torch.cat(xs, 0)]).cuda()

    win_joint_latend_walk_1 = vis.images(xs, 7, 2, opts={'caption': 'win_joint_latend_walk_1'},
                                         win=win_joint_latend_walk_1)
    #patterm2
    """zs = zs[0:2]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-3, 3, 7), volatile=True).type_as(zs)
    delta_1 = torch.autograd.Variable(torch.linspace(3, -3, 7), volatile=True).type_as(zs)

    for i in range(0, z_dim, gap):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        vec[:, i + 1] = 1
        vec = vec * delta_1[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_delta[:, :, i + 1] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = torch.tensor([item.cpu().detach().numpy() for item in torch.cat(xs, 0)]).cuda()

    win_joint_latend_walk_2 = vis.images(xs, 7, 2, opts={'caption': 'win_joint_latend_walk_2'},
                                         win=win_joint_latend_walk_2)"""





def plot_elbo(train_elbo,mu_display,tc_display,add_display,rec_display, vis):
    global win_train_elbo,win_TC,win_MU,win_add,win_REC
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)
    win_MU = vis.line(torch.Tensor(mu_display), opts={'markers': True}, win=win_MU)
    win_TC = vis.line(torch.Tensor(tc_display), opts={'markers': True}, win=win_TC)
    win_add = vis.line(torch.Tensor(add_display), opts={'markers': True}, win=win_add)
    win_REC = vis.line(torch.Tensor(rec_display), opts={'markers': True}, win=win_REC)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def draw_heatmap(map,path,iter,try_time,beta_range_dict,args):

    non_zero = map.nonzero()
    im_min = map[non_zero].min()

    max_index = np.unravel_index(np.argmax(map, axis=None), map.shape)
    im_max = map[max_index]
    if args.tcvae ==True:
        vmin = im_min
        vmax = im_max
        if iter in beta_range_dict.keys():
            total_min = beta_range_dict[iter]["min"]*try_time
            beta_range_dict[iter].update({"min":(total_min+im_min)/(try_time+1)})
            total_max = beta_range_dict[iter]["max"] * try_time
            beta_range_dict[iter].update({"max": (total_max + im_max) / (try_time + 1)})
        else:
            beta_range_dict.update({iter: {"min": im_min,"max": im_max}})
    elif args.newTC ==True:
        vmax = beta_range_dict[iter]["max"]
        vmin = beta_range_dict[iter]["min"]

    fig = plt.figure()
    mask = np.zeros_like(map, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns_plot = sns.heatmap(map,mask = mask,annot=True,cmap="YlGnBu",vmin=min(vmin,im_min),vmax=max(im_min,vmax),annot_kws={"fontsize":6})
    #plt.show()
    f = plt.gcf()  
    f.savefig(path+'\\'+str(iter)+'_'+str(try_time)+'.png',dpi=1000,bbox_inches = 'tight')
    f.clear()  

def main(args,train_loader):
    visdom_env = "/test"

    # parse command line arguments
    torch.cuda.set_device(args.gpu)
    # data loader
    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss , newTC=args.newTC,contraintInfo = args.contraintInfo,nur_num = args.nur_num,layer = args.layer)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env=visdom_env, port=6006)
    detail_dict = {}
    train_elbo = []
    tc_display = []
    real_tc_display = []
    fake_tc_display = []
    mu_display = []
    add_display = []
    rec_display = []
    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = 100
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    #print("需要运行",num_iterations)
    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
            if width == 28:
                x = x[0]
            torch.set_printoptions(profile="full")
            #print(x)
            x = x.cuda()
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo ,mu,tc,real_tc,fake_tc,add,rec,headMap,mutalInfo = vae.elbo(x,dataset_size,args.split_num,args.beta)

            if utils.isnan(obj).any():
                continue
                #raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()
            """print("[iteration %03d] ELBO: %.4f (%.4f)"% (
                    iteration,elbo_running_mean.val, elbo_running_mean.avg))"""
            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                mu_display.append(mu.sum().item())
                tc_display.append(tc.sum().item())
                real_tc_display.append(real_tc.item())
                fake_tc_display.append(fake_tc.item())
                add_display.append(add.sum().item())
                rec_display.append(rec.mean().item())
                """print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                    iteration, time.time() - batch_time, vae.beta, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg))"""

                vae.eval()

                # plot training and test ELBOs
                if args.visdom:
                    display_samples(vae, x, vis,args.split_num)
                    #plot_elbo(train_elbo, mu_display,tc_display,add_display,rec_display, vis)

                """utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, visdom_env, 0)"""
                """eval('plot_vs_gt_' + args.dataset)(vae, train_loader.dataset,
                    os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))"""
            if iteration>num_iterations:
                break
    #draw_heatmap(headMap,visdom_env,iter,try_time,beta_range_dict,args)
    detail_dict.update({"train_elbo":train_elbo,"tc_display":tc_display,"real_tc":real_tc_display,"fake_tc":fake_tc_display,"mu_display":mu_display,"add_display":add_display,"rec_display":rec_display})
    # Report statistics after training
    """vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, 0)
    dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
    eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))"""
    return detail_dict,vae


if __name__ == '__main__':
    state_dir = "/3dshapes/"
    result = {}
    arg_list = []
    beta_range_dict = {}
    MYbeta = 10
    #arg_list.append(args3)
    nur_num_list = [1000]
    #####################################################################################################################
    for ii in range(50):
        for latend_dim in [16]:
            layer = 2
            for nur_num in nur_num_list:
                contraintInfo = 2
                split_num = 2
                parser3 = argparse.ArgumentParser(description="parse args")
                parser3.add_argument('-d', '--dataset', default='shape3d', type=str, help='dataset name',
                                     choices=["shape3d","cars3d",'shapes', 'faces', 'minest'])
                parser3.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
                parser3.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
                parser3.add_argument('-b', '--batch-size', default=128, type=int, help='batch size')
                parser3.add_argument('-l', '--learning-rate', default=1e-4, type=float, help='learning rate')
                parser3.add_argument('-z', '--latent_dim', default=latend_dim, type=int,
                                     help='size of latent dimension')
                parser3.add_argument('--beta', default=MYbeta, type=float, help='ELBO penalty term')
                parser3.add_argument('--tcvae', default=False, action='store_true')
                parser3.add_argument('--exclude-mutinfo', action='store_true')
                parser3.add_argument('--beta-anneal', action='store_true')
                parser3.add_argument('--lambda-anneal', action='store_true')
                parser3.add_argument('--mss', default=True, action='store_true',
                                     help='use the improved minibatch estimator')
                parser3.add_argument('--conv', action='store_true')
                parser3.add_argument('--gpu', type=int, default=0)
                parser3.add_argument('--visdom', default=False, action='store_true',
                                     help='whether plotting in visdom is desired')
                parser3.add_argument('--contraintInfo', type=int, default=contraintInfo)
                parser3.add_argument('--log_freq', default=100, type=int, help='num iterations per log')
                parser3.add_argument('--newTC', default=True, action='store_true', help='new alg')
                parser3.add_argument('--split_num', type=int, default=split_num)
                parser3.add_argument('--nur_num', type=int, default=nur_num)
                parser3.add_argument('--layer', type=int, default=layer)
                parser3.add_argument('--save', type=str, default="/root/VAE/3dshape/new_betaTCvae_3dshape_cov/result/")
                # parser3.add_argument('--testNmae',type=)
                args3 = parser3.parse_args()

                train_loader = setup_data_loaders(args3, use_cuda=True)
                iter_dict,vae = main(args3,train_loader)
                torch.save(vae.state_dict(), state_dir + '_stcvae_'+args3.dataset+'_'+str(ii)+'.pth')