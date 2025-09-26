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

"""Tests for factor_vae.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import shapes3d,cars3d,dsprites,mpi3d
from disentanglement_lib.evaluation.metrics import factor_vae,sap_score
import numpy as np
import model
import model_CONTRL
import torch
import json
import gc
class FactorVaeTest(absltest.TestCase):

  def test_metric_contrl(self):#集合GuidedVAE在所有训练集下的FactorVAE评分
    total_scores_3dshapes = []
    total_scores_3dcars = []
    total_scores_dsprites = []
    netType = "contrl" #'contrl' 'guide'
    #baseRoot = "E:/python3.8.10/factor_Guided/unGuidedVAE_"
    baseRoot = "E:/python3.8.10/ContrlVAE/contrlVAE_"
    
    for i in range(100):
      ground_truth_data = mpi3d.MPI3D()
      if netType =="guide":
        VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =3,raw_liner_dim = 12288,n_vae_dis = 8)
      elif netType =="contrl":
        VAE_model =  model_CONTRL.BetaVAE_H(z_dim=8, nc=3)
      VAE_model.load_state_dict(torch.load(baseRoot+ "3dmpi_"+str(i)+".pth"))
      representation_function = VAE_model.function_z
      #representation_function = lambda x: x
      random_state = np.random.RandomState(0)
      scores = factor_vae.compute_factor_vae(
          ground_truth_data, representation_function, random_state, None, 5, 3000,
          2000, 2500)
      print("scores",scores)
      total_scores_3dshapes.append(scores)
      VAE_model=None
      ground_truth_data = None
      torch.cuda.empty_cache()
      gc.collect()
    with open(baseRoot+"3dmpi_factor_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dshapes, f)


  def test_metric_guide(self):#集合GuidedVAE在所有训练集下的FactorVAE评分
    total_scores_3dshapes = []
    total_scores_3dcars = []
    total_scores_dsprites = []
    netType = "contrl" #'contrl' 'guide'
    #baseRoot = "E:/python3.8.10/factor_Guided/unGuidedVAE_"
    baseRoot = "E:/python3.8.10/ContrlVAE/contrlVAE_"
    
    for i in range(100):
      ground_truth_data = mpi3d.MPI3D()
      if netType =="guide":
        VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =3,raw_liner_dim = 12288,n_vae_dis = 8)
      elif netType =="contrl":
        VAE_model =  model_CONTRL.BetaVAE_H(z_dim=8, nc=3)
      VAE_model.load_state_dict(torch.load(baseRoot+ "3dmpi_"+str(i)+".pth"))
      representation_function = VAE_model.function_z
      #representation_function = lambda x: x
      random_state = np.random.RandomState(0)
      scores = sap_score.compute_sap(
          ground_truth_data, representation_function, random_state, 1,None, 3000,
          3000, continuous_factors=False)
      print("scores",scores)
      total_scores_3dshapes.append(scores)
      VAE_model=None
      ground_truth_data = None
      torch.cuda.empty_cache()
      gc.collect()
    with open(baseRoot+"3dmpi_sap_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dshapes, f)


def factor_main():
  absltest.main()

if __name__ == "__main__":
  absltest.main()
