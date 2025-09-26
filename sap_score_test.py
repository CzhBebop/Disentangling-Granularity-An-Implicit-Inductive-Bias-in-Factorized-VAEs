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

"""Tests for sap_score.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import shapes3d,cars3d,dsprites
from disentanglement_lib.evaluation.metrics import sap_score
import numpy as np
import model
import model_CONTRL
import torch
import json
import gc
class SapScoreTest(absltest.TestCase):

  def test_metric_contrl(self):
    total_scores_3dshapes = []
    total_scores_3dcars = []
    total_scores_dsprites = []

    #baseRoot = "E:/python3.8.10/factor_Guided/unGuidedVAE_"
    baseRoot = "E:/python3.8.10/ContrlVAE/contrlVAE_"
    netType = "contrl" #"guide" "contrl"



    for i in range(100):

      ground_truth_data = shapes3d.Shapes3D()
      if netType =="guide":
        VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =3,raw_liner_dim = 12288,n_vae_dis = 8)
      elif netType =="contrl":
        VAE_model =  model_CONTRL.BetaVAE_H(z_dim=8, nc=3)
      VAE_model.load_state_dict(torch.load(baseRoot+ "3dshapes_"+str(i)+".pth"))
      representation_function = VAE_model.function_z
      random_state = np.random.RandomState(0)
      scores = sap_score.compute_sap(
          ground_truth_data, representation_function, random_state, 1,None, 3000,
          3000, continuous_factors=False)
      #self.assertBetween(scores["SAP_score"], 0.9, 1.0)
      print("sap",scores)
      total_scores_3dshapes.append(scores)
      VAE_model=None
      ground_truth_data = None
      torch.cuda.empty_cache()
      gc.collect()
    with open(baseRoot+"3dshapes_SAP_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dshapes, f)

    
    """for i in range(1):
      ground_truth_data = cars3d.Cars3D()
      if netType =="guide":
        VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =3,raw_liner_dim = 12288,n_vae_dis = 8)
      elif netType =="contrl":
        VAE_model =  model_CONTRL.BetaVAE_H(z_dim=8, nc=3)
      VAE_model.load_state_dict(torch.load(baseRoot+"3dcars_"+str(i)+".pth"))
      representation_function = VAE_model.function_z
      random_state = np.random.RandomState(0)
      scores = sap_score.compute_sap(
          ground_truth_data, representation_function, random_state, None, 3000,
          3000, continuous_factors=True)
      #self.assertBetween(scores["SAP_score"], 0.9, 1.0)
      print("sap",scores)
      total_scores_3dcars.append(scores)
    with open(baseRoot+ "3dcars_SAP_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dcars, f)
    

    
    for i in range(1):
      ground_truth_data = dsprites.DSprites()
      if netType =="guide":
        VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =1,raw_liner_dim = 4096,n_vae_dis = 8)
      elif netType =="contrl":
        VAE_model =  model_CONTRL.BetaVAE_H(z_dim=8, nc=1)
      VAE_model.load_state_dict(torch.load(baseRoot+"dsprites_"+str(i)+".pth"))
      representation_function = VAE_model.function_z
      random_state = np.random.RandomState(0)
      scores = sap_score.compute_sap(
          ground_truth_data, representation_function, random_state, None, 3000,
          3000, continuous_factors=True)
      #self.assertBetween(scores["SAP_score"], 0.9, 1.0)
      print("sap",scores)
      total_scores_dsprites.append(scores)
    with open(baseRoot+"dsprites_SAP_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_dsprites, f)"""

      
  def test_metric_guide(self):
    total_scores_3dshapes = []
    total_scores_3dcars = []
    total_scores_dsprites = []

    baseRoot = "E:/python3.8.10/factor_Guided/unGuidedVAE_"
    #baseRoot = "E:/python3.8.10/ContrlVAE/contrlVAE_"
    netType = "guide" #"guide" "contrl"



    for i in range(100):

      ground_truth_data = shapes3d.Shapes3D()
      if netType =="guide":
        VAE_model = model.unGuidedVAE(raw_width =64,raw_channel =3,raw_liner_dim = 12288,n_vae_dis = 8)
      elif netType =="contrl":
        VAE_model =  model_CONTRL.BetaVAE_H(z_dim=8, nc=3)
      VAE_model.load_state_dict(torch.load(baseRoot+ "3dshapes_"+str(i)+".pth"))
      representation_function = VAE_model.function_z
      random_state = np.random.RandomState(0)
      scores = sap_score.compute_sap(
          ground_truth_data, representation_function, random_state, 1,None, 3000,
          3000, continuous_factors=False)
      #self.assertBetween(scores["SAP_score"], 0.9, 1.0)
      print("sap",scores)
      total_scores_3dshapes.append(scores)
      VAE_model=None
      ground_truth_data = None
      torch.cuda.empty_cache()
      gc.collect()
    with open(baseRoot+"3dshapes_SAP_RESLUT_.json", 'w') as f:
                  json.dump(total_scores_3dshapes, f)









  """def test_bad_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: np.zeros_like(x, dtype=np.float64)
    random_state = np.random.RandomState(0)
    scores = sap_score.compute_sap(
        ground_truth_data, representation_function, random_state, None, 3000,
        3000, continuous_factors=True)
    self.assertBetween(scores["SAP_score"], 0.0, 0.2)

  def test_duplicated_latent_space(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = sap_score.compute_sap(
        ground_truth_data, representation_function, random_state, None, 3000,
        3000, continuous_factors=True)
    self.assertBetween(scores["SAP_score"], 0.0, 0.2)"""



def sap_main():
  absltest.main()
if __name__ == "__main__":
  absltest.main()
