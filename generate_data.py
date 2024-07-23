#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:19:11 2024

@author: fmry
"""

#%% Modules

import jax.numpy as jnp
import jax.random as jrandom

from jax import vmap

import os

#argparse
import argparse

#%% Arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data', default="Circle3D",
                        type=str)
    parser.add_argument('--N_data', default=50000,
                        type=int)
    parser.add_argument('--std_noise', default=0.01,
                        type=float)
    parser.add_argument('--data_path', default="data/",
                        type=str)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% Generate data

def generate_data():
   
    args = parse_args()
    
    save_path = ''.join((args.data_path, args.data, '/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if args.data == "Circle3D":
        rng_key = jrandom.PRNGKey(args.seed)
        noise_level = args.std_noise*jrandom.normal(rng_key, shape=(3,args.N_data))
        
        theta = jrandom.uniform(rng_key, shape=(args.N_data,), minval=0.0, maxval=2.0*jnp.pi)
        x1 = jnp.cos(theta)
        x2 = jnp.sin(theta)
        x3 = 1.0*jnp.ones_like(x1)
        
        X = (jnp.vstack((x1,x2,x3))+noise_level).T
        
        data_path = ''.join((save_path, 'data.npy'))
        with open(data_path, 'wb') as f:
            jnp.save(f, X)
    
    return

#%% Main

if __name__ == '__main__':
        
    generate_data()