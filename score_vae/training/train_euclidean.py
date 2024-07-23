#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources


#%% Modules

from score_vae.setup import *

from score_vae.loader import save_model
from score_vae.loss_fun import elbo_euclidean

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val: Dict
    opt_state: optax.OptState
    rng_key: Array
    
#%% Train Euclidean Brownian VAE

def train_euclidean(vae_model:object,
                    data_generator:object,
                    lr_rate:float = 0.0002,
                    save_path:str = '',
                    split:float = 0.0,
                    batch_size:int = 100,
                    vae_state:TrainingState = None,
                    epochs:int=100,
                    save_step:int = 100,
                    vae_optimizer:object = None,
                    seed:int = 2712
                    )->None:
    
    @partial(jit, static_argnames=['training_type'])
    def vae_loss(params:Dict, state:TrainingState, data:Array, rng_key:Array, training_type="All"):
        
        vae_output = vae_apply_fn(params, data, state.rng_key, state.state_val)
        
        return elbo_euclidean(data, vae_output, training_type)
    
    @partial(jit, static_argnames=['training_type'])
    def update(state:TrainingState, data:Array, training_type="All"):
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(vae_loss, has_aux=True)(state.params,
                                                                 state, 
                                                                 data, 
                                                                 rng_key,
                                                                 training_type=training_type)
        updates, new_opt_state = vae_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, next_rng_key), loss
    
    if vae_optimizer is None:
        vae_optimizer = optax.adam(learning_rate = lr_rate,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(vae_model) == hk.Transformed:
        if vae_state is None:
            initial_params = vae_model.init(jrandom.PRNGKey(seed), next(data_generator))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        if vae_state is None:
            initial_params, init_state = vae_model.init(jrandom.PRNGKey(seed), next(data_generator))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, state_val, rng_key, data)[0]
    
    if split>0.0:
        epochs_encoder = int(split*epochs)
        epochs_decoder = int((1-split)*epochs)
        epochs = 0
    else:
        epochs_encoder = 0
        epochs_decoder = 0

    for step in range(epochs_encoder):
        vae_state, loss = update(vae_state, next(data_generator), training_type="Encoder")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    for step in range(epochs_decoder):
        vae_state, loss = update(vae_state, next(data_generator), training_type="Decoder")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    for step in range(epochs):
        vae_state, loss = update(vae_state, next(data_generator), training_type="All")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
          
    save_model(save_path, vae_state)
    
    return


