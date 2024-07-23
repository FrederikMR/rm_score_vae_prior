#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:44:54 2024

@author: fmry
"""

#%% Modules

from score_vae.setup import *

#%% Load numpy dataset

def load_tensor_slices(X:Array, batch_size:int=100, seed:int=2712):
    
    return tf.data.Dataset.from_tensor_slices(X).shuffle(buffer_size=10 * batch_size, seed=seed)\
        .batch(batch_size) \
            .prefetch(buffer_size=5) \
                .repeat() \
                    .as_numpy_iterator()