#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:25:28 2024

@author: fmry
"""

#%% Sources

#%% Modules

from score_vae.setup import *

#%% Denoising Score Matching First Order

@partial(jit, static_argnames=['score_model'])
def dsm(x0:Array,
        xt:Array,
        t:Array,
        dW:Array,
        dt:Array,
        score_model:Callable[[Array, Array, Array], Array],
        )->Array:

    s1 = score_model(x0,xt,t)
    
    return jnp.mean(jnp.sum(dW/dt+s1, axis=-1))
    
#%% Variance Reduction Denoising Score Matching First Order

@partial(jit, static_argnames=['score_model'])
def dsmvr(x0:Array,
          xt:Array,
          t:Array,
          dW:Array,
          dt:Array,
          score_model:Callable[[Array,Array,Array], Array]
          )->Array:
    
    s1 = score_model(x0,xt,t)
    s1p = score_model(x0,x0,t)
    
    z = dW/dt
    
    l1_loss = z+s1
    l1_loss = 0.5*jnp.einsum('...j,...j->...', l1_loss)
    var_loss = jnp.einsum('...j,...j->...', s1p, z)+jnp.einsum('...j,...j->...', z, z)
    
    return jnp.mean(l1_loss-var_loss)