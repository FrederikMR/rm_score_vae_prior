#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources


#%% Modules

#jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, lax, vmap, jacfwd, value_and_grad
from jax import tree_leaves, tree_map, tree_flatten, tree_unflatten

#numpy
import numpy as np

#haiku
import haiku as hk

#optax
import optax

#tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

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
from typing import Callable, Tuple, NamedTuple

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
    opt_state: optax.OptState
    rng_key: Array
    
#%% Pre-Train VAE

def pretrain_vae(vae_model:object,
                 data_generator:object,
                 lr_rate:float = 0.002,
                 save_path:str = '',
                 split:float = 0.0,
                 batch_size:int=100,
                 vae_state:TrainingState = None,
                 epochs:int=1000,
                 save_step:int = 100,
                 vae_optimizer:object = None,
                 seed:int=2712,
                 )->None:
    
    @partial(jit, static_argnames=['training_type'])
    def update(state:TrainingState, data:Array, training_type="All"):
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(vae_euclidean_loss, has_aux=True)(state.params, state.state_val, state.rng_key, data,
                                             vae_apply_fn, training_type=training_type)
        updates, new_opt_state = vae_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
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
            initial_params = vae_model.init(jrandom.PRNGKey(seed), 
                                            next(data_generator.batch(batch_size).repeat().as_numpy_iterator()))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        if vae_state is None:
            initial_params, init_state = vae_model.init(jrandom.PRNGKey(seed), 
                                                        next(data_generator.batch(batch_size).repeat().as_numpy_iterator()))
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
    
    #print(vae_state.params['vaebm/~muz/prior_layer'])
    #print(vae_state.params['vaebm/~tz/prior_layer'])
    for step in range(epochs_encoder):
        dataset_epoch = data_generator.batch(batch_size)
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="Encoder")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    for step in range(epochs_decoder):
        dataset_epoch = data_generator.batch(batch_size)
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="Decoder")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    
    for step in range(epochs):
        dataset_epoch = data_generator.batch(batch_size)
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="All")
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    #print(vae_state.params['vaebm/~muz/prior_layer'])
    #print(vae_state.params['vaebm/~tz/prior_layer'])
          
    save_model(save_path, vae_state)
    
    return

#%% Pre-Train Scores

def pretrain_scores(score_model:object,
                    vae_state:object,
                    decoder_model:object,
                    x0s:Array,
                    lr_rate:float = 0.002,
                    save_path:str = '',
                    repeats:int=8,
                    x_samples:int=2**5,
                    t_samples:int=2**7,
                    max_T:float=1.0,
                    dt_steps:int=1000,
                    training_type:str = 'dsmvr',
                    score_state:TrainingState = None,
                    epochs:int=1000,
                    save_step:int = 100,
                    score_optimizer:object = None,
                    seed:int=2712,
                    ):
    
    @jit
    def loss_fun(params:hk.Params,  state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: score_apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:dim]
        xt = data[:,dim:(2*dim)]
        t = data[:,2*dim]
        dW = data[:,(2*dim+1):-1]
        dt = data[:,-1]
        
        return loss_model(score_generator, s1_model, params, state_val, rng_key,
                          x0, xt, t, dW, dt)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if training_type == "vsm":
        loss_model = vsm_s1fun
    elif training_type == "dsm":
        loss_model = dsm_s1fun
    elif training_type == "dsmvr":
        loss_model = dsmvr_s1fun
    else:
        raise Exception("Invalid loss type. You can choose: vsm, dsm, dsmvr")
        return
    
    initial_rng_key = jrandom.PRNGKey(seed)
    if score_optimizer is None:
        score_optimizer = optax.adam(learning_rate = lr_rate,
                                     b1 = 0.9,
                                     b2 = 0.999,
                                     eps = 1e-08,
                                     eps_root = 0.0,
                                     mu_dtype=None)
        
    batch_size = x_samples*repeats*t_samples
    N_sim = x_samples*repeats
    F = lambda z: decoder_model.apply(vae_state.params, vae_state.rng_key, z.reshape(1,-1))

    dim = x0s.shape[-1]
    score_generator = VAESampling(F=F,
                                  x0=x0s,
                                  repeats=repeats,
                                  x_samples=x_samples,
                                  t_samples=t_samples,
                                  N_sim=N_sim,
                                  max_T=max_T,
                                  dt_steps=dt_steps,
                                  )
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
    
    if type(score_model) == hk.Transformed:
        if score_state is None:
            initial_params = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, rng_key, data)
    elif type(score_model) == hk.TransformedWithState:
        if score_state is None:
            initial_params, init_state = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, state_val, rng_key, data)[0]
        
    for step in range(epochs):
        data = next(score_datasets)
        if jnp.isnan(jnp.sum(data)):
            score_generator.x0s = score_generator.x0s_default
            score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,3*dim+2]))
            score_datasets = iter(tfds.as_numpy(score_datasets))
            continue
        score_new_state,loss = update_score(score_state, data)
        if ((not any(jnp.sum(jnp.isnan(val))>0 for val in score_new_state.params[list(score_new_state.params.keys())[0]].values())) \
                and (loss<1e12)):
            score_state = score_new_state
        else:
            score_generator.x0s = score_generator.x0s_default
            score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                           output_shapes=([batch_size,3*dim+2]))
            score_datasets = iter(tfds.as_numpy(score_datasets))
        if (step+1) % save_step == 0:            
            save_model(save_path, score_state)
            print("Epoch: {}".format(step+1))
    
    return

#%% Score Training

def train_vaebm(vae_model:object,
                decoder_model:object,
                score_model:object,
                vae_datasets:object,
                dim:int,
                vae_batch_size:int=100,
                epochs:int=1000,
                vae_epochs:int=100,
                score_epochs:int=100,
                vae_split:float=0.0,
                lr_rate_vae:float=0.002,
                lr_rate_score:float=0.002,
                score_repeats:int=8,
                score_x_samples:int=2**5,
                score_t_samples:int=2**7,
                max_T:float=1.0,
                dt_steps:int=1000,
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
    
    #@partial(jit, static_argnames=['training_type'])
    def update_vae(state:TrainingState, data:Array, training_type="All"):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(vae_riemannian_loss)(state.params, state.state_val, state.rng_key, data,
                                                        vae_apply_fn, score_apply_fn, score_state, 
                                                        training_type=training_type)
        updates, new_opt_state = vae_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    @jit
    def score_loss(params:hk.Params,  state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: score_apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
    
        x0 = data[:,:dim]
        xt = data[:,dim:(2*dim)]
        t = data[:,2*dim]
        dW = data[:,(2*dim+1):-1]
        dt = data[:,-1]
        
        return loss_model(score_generator, s1_model, params, state_val, rng_key,
                          x0, xt, t, dW, dt)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(score_loss)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if score_type == "vsm":
        loss_model = vsm_s1fun
    elif score_type == "dsm":
        loss_model = dsm_s1fun
    elif score_type == "dsmvr":
        loss_model = dsmvr_s1fun
    else:
        raise Exception("Invalid loss type. You can choose: vsm, dsm, dsmvr")
        return
        
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
        
    F = lambda z: decoder_apply_fn(vae_state.params, z.reshape(-1,dim), vae_state.rng_key, 
                                   vae_state.state_val)[0].reshape(-1)
    batch_size = score_x_samples*score_repeats*score_t_samples
    score_N_sim = score_x_samples*score_repeats
    x0 = jnp.zeros(dim)
    score_generator = VAESampling(F=F,
                                  x0=x0,
                                  repeats=score_repeats,
                                  x_samples=score_x_samples,
                                  t_samples=score_t_samples,
                                  N_sim=score_N_sim,
                                  max_T=1.0,
                                  dt_steps=dt_steps,
                                  )
    
    score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                   output_shapes=([batch_size,3*dim+2]))
    score_datasets = iter(tfds.as_numpy(score_datasets))
        
    initial_rng_key = jrandom.PRNGKey(seed)
    if type(vae_model) == hk.Transformed:
        if vae_state is None:
            initial_params = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets.batch(vae_batch_size)))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        if vae_state is None:
            initial_params, init_state = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets.batch(vae_batch_size)))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, state_val, rng_key, data)[0]
        
    decoder_apply_fn = lambda params, data, rng_key, state_val: decoder_model.apply(params, rng_key, data)
        
    if type(score_model) == hk.Transformed:
        if score_state is None:
            initial_params = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, rng_key, data)
    elif type(score_model) == hk.TransformedWithState:
        if score_state is None:
            initial_params, init_state = score_model.init(jrandom.PRNGKey(seed), next(score_datasets)[:,:(2*dim+1)])
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, state_val, rng_key, data)[0]
        
    if vae_split>0:
        epochs_encoder = int(vae_split*vae_epochs)
        epochs_decoder = int((1-vae_split)*vae_epochs)
        vae_epochs = 0
    else:
        epochs_encoder = 0
        epochs_decoder = 0
    
    for step in range(epochs):
        for vae_step in range(epochs_encoder):
            dataset_epoch = vae_datasets.batch(vae_batch_size)
            for ds in dataset_epoch:
                vae_state, loss = update_vae(vae_state, jnp.array(ds), training_type="Encoder")
        for vae_step in range(epochs_decoder):
            dataset_epoch = vae_datasets.batch(vae_batch_size)
            for ds in dataset_epoch:
                vae_state, loss = update_vae(vae_state, jnp.array(ds), training_type="Decoder")
        for vae_step in range(vae_epochs):
            dataset_epoch = vae_datasets.batch(vae_batch_size)
            for ds in dataset_epoch:
                vae_state, loss = update_vae(vae_state, jnp.array(ds), training_type="All")
                print(loss)
        save_model(vae_path, vae_state)

        F = lambda z: decoder_apply_fn(vae_state.params, z.reshape(-1,dim), vae_state.rng_key, 
                                       vae_state.state_val)[0].reshape(-1)
        z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z = vae_apply_fn(params, x, rng_key, state_val)
        score_generator.F = F
        z0 = z[np.round(np.linspace(0, len(z) - 1, score_repeats)).astype(int)]
        score_generator.x0s = z0
        score_generator.max_T = jnp.maxmimum(2*jnp.exp(2*log_t_z[0]),1.0).squeeze()
        score_generator.x0s_default = z0
        score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                       output_shapes=([batch_size,3*dim+2]))
        score_datasets = iter(tfds.as_numpy(score_datasets))
        
        for score_step in range(score_epochs):
            data = next(score_datasets)
            if jnp.isnan(jnp.sum(data)):
                score_generator.x0s = score_generator.x0s_default
                score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                               output_shapes=([batch_size,3*dim+2]))
                score_datasets = iter(tfds.as_numpy(score_datasets))
                continue
            score_new_state,loss = update_score(score_state, data)
            if ((not any(jnp.sum(jnp.isnan(val))>0 for val in score_new_state.params[list(score_new_state.params.keys())[0]].values())) \
                    and (loss<1e12)):
                score_state = score_new_state
            else:
                score_generator.x0s = score_generator.x0s_default
                score_datasets = tf.data.Dataset.from_generator(score_generator,output_types=tf.float32,
                                                               output_shapes=([batch_size,3*dim+2]))
                score_datasets = iter(tfds.as_numpy(score_datasets))
        save_model(score_path, score_state)
        if (step+1) % save_step == 0:
            print("Epoch: {}".format(step+1))
          
    save_model(score_path, score_state)
    save_model(vae_path, vae_state)
    
    return
