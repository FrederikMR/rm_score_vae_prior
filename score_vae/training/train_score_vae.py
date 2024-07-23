#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources

#%% Modules

from score_vae.setup import *

from score_vae.loss_fun import dsm, dsmvr, elbo_scores
from score_vae.loader import save_model

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  Dict
    opt_state: optax.OptState
    rng_key: Array

#%% Train Score VAE    

def train_score_vae(vae_model:object,
                    score_model:object,
                    data_generator:object,
                    dim:int,
                    batch_size:int=100,
                    epochs:int=1000,
                    vae_split:float=0.0,
                    lr_rate_vae:float=0.002,
                    lr_rate_score:float=0.002,
                    vae_optimizer:object = None,
                    score_optimizer:object = None,
                    vae_state:TrainingState=None,
                    score_state:TrainingState=None,
                    seed:int=2712,
                    save_step:int=100,
                    score_type:str='dsmvr',
                    vae_path:str='',
                    score_path:str='',
                    )->None:
    
    @partial(jit, static_argnames=['training_type'])
    def vae_loss(params:hk.Params, vae_state:TrainingState, score_state:TrainingState, rng_key:Array, data:Array, training_type="All"):
        
        vae_output = vae_apply_fn(params, data, rng_key, vae_state.state_val)
        
        t_zx = jnp.exp(vae_output.log_t_zx)
        t_z = jnp.exp(vae_output.log_t_z)
        
        prior_data = jnp.hstack((vae_output.mu_z, vae_output.z, t_z.reshape(-1,1)))
        post_data = jnp.hstack((vae_output.mu_zx, vae_output.z, t_zx.reshape(-1,1)))
        
        s_log_qzx = score_apply_fn(score_state.params, post_data, score_state.rng_key, score_state.state_val)
        s_log_pz = score_apply_fn(score_state.params, prior_data, score_state.rng_key, score_state.state_val)
        
        return elbo_scores(data, vae_output, s_log_qzx, s_log_pz, training_type)
    
    @jit
    def score_loss(params:hk.Params, vae_state:TrainingState, score_state:TrainingState, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: score_apply_fn(params, jnp.hstack((x,y,t.reshape(-1,1))), rng_key, score_state.state_val)
        vae_output = vae_apply_fn(vae_state.params, data, vae_state.rng_key, vae_state.state_val)
        
        t_zx = jnp.exp(vae_output.log_t_zx)
        t_z = jnp.exp(vae_output.log_t_z)
        
        x0 = jnp.vstack((vae_output.mu_zx, vae_output.mu_z))
        xt = jnp.vstack((vae_output.z, vae_output.z_prior))
        t = jnp.hstack((t_zx, t_z))
        dW = jnp.vstack((vae_output.dW_zx, vae_output.dW_z))
        dt = jnp.vstack((vae_output.dt_zx.reshape(-1,1), vae_output.dt_z.reshape(-1,1)))
        
        return sloss_model(x0, xt, t, dW, dt, s1_model)
    
    @partial(jit, static_argnames=['training_type'])
    def update(vae_state:TrainingState, score_state:TrainingState, data:Array, training_type="All"):

        rng_key, next_rng_key = jrandom.split(score_state.rng_key)
        s_loss, gradients = value_and_grad(score_loss)(score_state.params, vae_state, score_state, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, score_state.opt_state)
        new_params = optax.apply_updates(score_state.params, updates)
        
        score_state = TrainingState(new_params, score_state.state_val, new_opt_state, next_rng_key)
        
        rng_key, next_rng_key = jrandom.split(vae_state.rng_key)
        v_loss, gradients = value_and_grad(vae_loss, has_aux=True)(vae_state.params, 
                                                                 vae_state, 
                                                                 score_state,
                                                                 rng_key,
                                                                 data, 
                                                                 training_type=training_type)
        updates, new_opt_state = vae_optimizer.update(gradients, vae_state.opt_state)
        new_params = optax.apply_updates(vae_state.params, updates)
        
        vae_state = TrainingState(new_params, vae_state.state_val, new_opt_state, next_rng_key)
        
        return (score_state, s_loss), (vae_state, v_loss)
    
    if score_type == "dsm":
        sloss_model = dsm
    elif score_type == "dsmvr":
        sloss_model = dsmvr
    else:
        raise ValueError("Invalid loss type. You can choose: vsm, dsm, dsmvr")
        
    if vae_optimizer is None:
        vae_optimizer = optax.adam(learning_rate = lr_rate_vae,
                               b1 = 0.9,
                               b2 = 0.999,
                               eps = 1e-08,
                               eps_root = 0.0,
                               mu_dtype=None)
        
    if score_optimizer is None:
        score_optimizer = optax.adam(learning_rate = lr_rate_score,
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
        
    if vae_split>0:
        epochs_encoder = int(vae_split*epochs)
        epochs_decoder = int((1-vae_split)*epochs)
        epochs = 0
    else:
        epochs_encoder = 0
        epochs_decoder = 0

    for step in range(epochs_encoder):
        score, vae = update(vae_state, score_state, next(data_generator), training_type="Encoder")
        score_state, vae_state = score[0], vae[0]
        if (step+1) % save_step == 0:
            s_loss, v_loss = score[1], vae[1]
            print(f"Epoch: {step+1} \t VAE Loss: {v_loss[0]:.4f} \t Rec Loss: {v_loss[1][0]:.4f} \t KLD Loss: {v_loss[1][1]:.4f}  \t Score Loss: {s_loss:.4f}")
            save_model(score_path, score_state)
            save_model(vae_path, vae_state)
    for step in range(epochs_decoder):
        score, vae = update(vae_state, score_state, next(data_generator), training_type="Decoder")
        score_state, vae_state = score[0], vae[0]
        if (step+1) % save_step == 0:
            s_loss, v_loss = score[1], vae[1]
            print(f"Epoch: {step+1} \t VAE Loss: {v_loss[0]:.4f} \t Rec Loss: {v_loss[1][0]:.4f} \t KLD Loss: {v_loss[1][1]:.4f}  \t Score Loss: {s_loss:.4f}")
            save_model(score_path, score_state)
            save_model(vae_path, vae_state)
    for step in range(epochs):
        score, vae = update(vae_state, score_state, next(data_generator), training_type="All")
        score_state, vae_state = score[0], vae[0]
        if (step+1) % save_step == 0:
            s_loss, v_loss = score[1], vae[1]
            print(f"Epoch: {step+1} \t VAE Loss: {v_loss[0]:.4f} \t Rec Loss: {v_loss[1][0]:.4f} \t KLD Loss: {v_loss[1][1]:.4f}  \t Score Loss: {s_loss:.4f}")
            save_model(score_path, score_state)
            save_model(vae_path, vae_state)
          
    save_model(score_path, score_state)
    save_model(vae_path, vae_state)
    
    return
