#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources


#%% Modules

from score_vae.setup import *

from score_vae.loss_fun import dsm, dsmvr
from score_vae.loader import save_model

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val: Dict
    opt_state: optax.OptState
    rng_key: Array

#%% Train Scores

def train_score(score_model:object,
                vae_model:object,
                vae_state:object,
                data_generator:object,
                dim:int,
                lr_rate:float = 0.002,
                save_path:str = '',
                batch_size:int=100,
                training_type:str = 'dsm',
                score_state:TrainingState = None,
                epochs:int=1000,
                save_step:int = 100,
                score_optimizer:object = None,
                seed:int=2712,
                ):

    @jit
    def score_loss(params:hk.Params, state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: score_apply_fn(params, jnp.hstack((x,y,t.reshape(-1,1))), rng_key, state_val)
        vae_output = vae_apply_fn(data)
        
        t_zx = jnp.exp(vae_output.log_t_zx)
        t_z = jnp.exp(vae_output.log_t_z)
        
        x0 = jnp.vstack((vae_output.mu_zx, vae_output.mu_z))
        xt = jnp.vstack((vae_output.z, vae_output.z_prior))
        t = jnp.hstack((t_zx, t_z))
        dW = jnp.vstack((vae_output.dW_zx, vae_output.dW_z))
        dt = jnp.vstack((vae_output.dt_zx.reshape(-1,1), vae_output.dt_z.reshape(-1,1)))
        
        return loss_model(x0, xt, t, dW, dt, s1_model)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(score_loss)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, next_rng_key), loss
    
    if training_type == "dsm":
        loss_model = dsm
    elif training_type == "dsmvr":
        loss_model = dsmvr
    else:
        raise ValueError("Invalid loss type. You can choose: vsm, dsm, dsmvr")
    
    initial_rng_key = jrandom.PRNGKey(seed)
    if score_optimizer is None:
        score_optimizer = optax.adam(learning_rate = lr_rate,
                                     b1 = 0.9,
                                     b2 = 0.999,
                                     eps = 1e-08,
                                     eps_root = 0.0,
                                     mu_dtype=None)
        
    vae_apply_fn = lambda z: vae_model.apply(vae_state.params, vae_state.rng_key, z.reshape(1,-1))
    
    if type(vae_model) == hk.Transformed:
        vae_apply_fn = lambda data: vae_model.apply(vae_state.params, vae_state.rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        vae_apply_fn = lambda data: vae_model.apply(vae_state.params, vae_state.state_val, vae_state.rng_key, data)[0]

    if type(score_model) == hk.Transformed:
        if score_state is None:
            initial_params = score_model.init(jrandom.PRNGKey(seed), 1.0*jnp.ones((2*batch_size,dim*2+1)))
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, rng_key, data)
    elif type(score_model) == hk.TransformedWithState:
        if score_state is None:
            initial_params, init_state = score_model.init(jrandom.PRNGKey(seed), 1.0*jnp.ones((2*batch_size,dim*2+1)))
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, state_val, rng_key, data)[0]

    for step in range(epochs):
        score_state, loss = update_score(score_state, next(data_generator))
        if (step+1) % save_step == 0:
            save_model(save_path, score_state)
            print(f"Epoch: {step+1} \t Loss: {loss:.4f}")
    
    return


