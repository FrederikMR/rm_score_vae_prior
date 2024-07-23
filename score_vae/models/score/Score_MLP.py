#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:46:53 2024

@author: fmry
"""

#%% Modules

from score_vae.setup import *

#%% Score Net

class Score_MLP(hk.Module):
    def __init__(self,
                 dim:int,
                 layers:List=[500,500,500,500,500],
                 act:List=[tanh,tanh,tanh,tanh,tanh],
                 init:hk.initializers = None,
                 )->None:
        super().__init__()
        
        self.dim = dim
        self.layers = layers
        self.act = act
        self.init = init
    
    def model(self)->object:
        
        model = []
        for layer,act in zip(self.layers, self.act):
            model.append(hk.Linear(layer, w_init=self.init, b_init=self.init))
            model.append(act)
            
        model.append(hk.Linear(self.dim, w_init=self.init, b_init=self.init))
        
        return hk.Sequential(model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        return self.model()(x)
        
        #x_new = x.T
        #x1 = x_new[:self.dim].T
        #x2 = x_new[self.dim:(2*self.dim)].T
        #t = x_new[-1]
        
        #shape = list(x.shape)
        #shape[-1] = 1
        #t = x_new[-1].reshape(shape)
            
        #grad_euc = (x1-x2)/t
      
        #return self.model()(x)+grad_euc