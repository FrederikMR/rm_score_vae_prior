#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:07:20 2024

@author: fmry
"""

#%% Sources

#%% Modules

from score_vae.setup import *

#%% VAE Output

class VAEOutput(NamedTuple):
  x: Array
  mu_xz:Array
  log_sigma_xz:Array
  z:Array
  z_prior:Array
  mu_zx:Array
  log_t_zx:Array
  mu_z:Array
  log_t_z:Array
  dW_zx:Array
  dW_z:Array
  dt_zx:Array
  dt_z:Array