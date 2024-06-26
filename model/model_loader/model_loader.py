#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:05:23 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, lax, vmap, jacfwd, tree_leaves, tree_map, tree_flatten, tree_unflatten

#numpy
import numpy as np

#random
import random

#functools
from functools import partial

#os
import os

#pickle
import pickle

#data types
from jax import Array
from typing import Callable, Tuple

#%% Save Model

def save_model(ckpt_dir, state):
    
    file_name = os.path.join(ckpt_dir, "arrays.npy")
    with open(file_name, "wb") as f:
        for x in tree_leaves(state):
            np.save(f, x, allow_pickle=False)
            
    tree_struct = tree_map(lambda t: 0, state)
    file_name = os.path.join(ckpt_dir, "tree.pkl")
    with open(file_name, "wb") as f:
        pickle.dump(tree_struct, f)
    
    return

#%% Load Model

def load_model(ckpt_dir):
    
    file_name = os.path.join(ckpt_dir, "tree.pkl")
    with open(file_name, "rb") as f:
        tree_struct = pickle.load(f)
        
    leaves, treedef = tree_flatten(tree_struct)
    file_name = os.path.join(ckpt_dir, "arrays.npy")
    with open(file_name, "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return tree_unflatten(treedef, flat_state)