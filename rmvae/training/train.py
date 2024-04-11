#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:20:21 2023

@author: fmry
"""

#%% Sources


#%% Modules

from rmvae.setup import *

#%% TrainingState

class TrainingState(NamedTuple):
    params: hk.Params
    state_val:  dict
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

    for step in range(epochs_encoder):
        dataset_epoch = data_generator.batch(batch_size)
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="Encoder")
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
            save_model(save_path, vae_state)
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    for step in range(epochs_decoder):
        dataset_epoch = data_generator.batch(batch_size)
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="Decoder")
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
            save_model(save_path, vae_state)
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    for step in range(epochs):
        dataset_epoch = data_generator.batch(batch_size)
        j = 0
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="All")
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
            save_model(save_path, vae_state)
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
          
    save_model(save_path, vae_state)
    
    return

#%% Train Riemannain Brownian VAE with Distance Approximation

def train_rm_distance(vae_model:object,
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

    for step in range(epochs_encoder):
        dataset_epoch = data_generator.batch(batch_size)
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="Encoder")
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
            save_model(save_path, vae_state)
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    for step in range(epochs_decoder):
        dataset_epoch = data_generator.batch(batch_size)
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="Decoder")
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
            save_model(save_path, vae_state)
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
    for step in range(epochs):
        dataset_epoch = data_generator.batch(batch_size)
        j = 0
        for ds in dataset_epoch:
            vae_state, loss = update(vae_state, jnp.array(ds), training_type="All")
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
            save_model(save_path, vae_state)
        if (step+1) % save_step == 0:
            save_model(save_path, vae_state)
            print(f"Epoch: {step+1} \t ELBO: {loss[0]:.4f} \t RecLoss: {loss[1][0]:.4f} \t KLD: {loss[1][1]:.4f}")
          
    save_model(save_path, vae_state)
    
    return

#%% Train Scores

def train_scores(score_model:object,
                 vae_model:object,
                 vae_state:object,
                 data_generator:object,
                 dim:int,
                 lr_rate:float = 0.002,
                 save_path:str = '',
                 batch_size:int=100,
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
        z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, dW, dt, z_prior, dW_prior, dt_prior = vae_apply_fn(vae_state.params, 
                                                                                                                   data, 
                                                                                                                   vae_state.rng_key, 
                                                                                                                   vae_state.state_val)
        
        t_zx = jnp.exp(2*log_t_zx)
        t_z = jnp.exp(2*log_t_z)
        
        x0 = jnp.vstack((mu_zx, mu_z))
        xt = jnp.vstack((z, z_prior))
        t = jnp.vstack((t_zx, t_z))
        dW = jnp.vstack((dW, dW_prior))
        dt = jnp.vstack((dt.reshape(-1,1), dt_prior.reshape(-1,1)))
        
        return loss_model(s1_model, x0, xt, t, dW, dt)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(loss_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
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
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
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

    for step in range(epochs):
        dataset_epoch = data_generator.batch(batch_size)
        for ds in dataset_epoch:
            score_state, loss = update_score(score_state, jnp.array(ds))
        if (step+1) % save_step == 0:
            save_model(save_path, score_state)
            print(f"Epoch: {step+1} \t Loss: {loss:.4f}")
    
    return

#%% Train Joint

def train_score_rmvae(vae_model:object,
                      score_model:object,
                      vae_datasets:object,
                      dim:int,
                      vae_batch_size:int=100,
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
    def update_vae(state:TrainingState, data:Array, training_type="All"):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(vae_riemannian_loss)(state.params, state.state_val, state.rng_key, data,
                                                        vae_apply_fn, score_apply_fn, score_state, 
                                                        training_type=training_type)
        updates, new_opt_state = vae_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    @jit
    def score_fun(params:hk.Params,  state_val:dict, rng_key:Array, data:Array):
        
        s1_model = lambda x,y,t: score_apply_fn(params, jnp.hstack((x,y,t)), rng_key, state_val)
        z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, dW, dt, z_prior, dW_prior, dt_prior = vae_apply_fn(vae_state.params, 
                                                                                                                   data, 
                                                                                                                   vae_state.rng_key, 
                                                                                                                   vae_state.state_val)
        
        t_zx = jnp.exp(2*log_t_zx)
        t_z = jnp.exp(2*log_t_z)
        
        x0 = jnp.vstack((mu_zx, mu_z))
        xt = jnp.vstack((z, z_prior))
        t = jnp.vstack((t_zx, t_z))
        dW = jnp.vstack((dW, dW_prior))
        dt = jnp.vstack((dt.reshape(-1,1), dt_prior.reshape(-1,1)))
        
        return loss_model(s1_model, x0, xt, t, dW, dt)
    
    @jit
    def update_score(state:TrainingState, data:Array):
        
        rng_key, next_rng_key = jrandom.split(state.rng_key)
        loss, gradients = value_and_grad(score_fun)(state.params, state.state_val, rng_key, data)
        updates, new_opt_state = score_optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainingState(new_params, state.state_val, new_opt_state, rng_key), loss
    
    if score_type == "dsm":
        loss_model = dsm
    elif score_type == "dsmvr":
        loss_model = dsmvr
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
            initial_params = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets.batch(vae_batch_size).repeat().as_numpy_iterator()))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, rng_key, data)
    elif type(vae_model) == hk.TransformedWithState:
        if vae_state is None:
            initial_params, init_state = vae_model.init(jrandom.PRNGKey(seed), next(vae_datasets.batch(vae_batch_size).repeat().as_numpy_iterator()))
            initial_opt_state = vae_optimizer.init(initial_params)
            vae_state = TrainingState(initial_params, init_state, initial_opt_state, initial_rng_key)
        vae_apply_fn = lambda params, data, rng_key, state_val: vae_model.apply(params, state_val, rng_key, data)[0]
        
    decoder_apply_fn = lambda params, data, rng_key, state_val: decoder_model.apply(params, rng_key, data)
        
    if type(score_model) == hk.Transformed:
        if score_state is None:
            initial_params = score_model.init(jrandom.PRNGKey(seed), 1.0*jnp.ones((2*vae_batch_size,dim*2+1)))
            initial_opt_state = score_optimizer.init(initial_params)
            score_state = TrainingState(initial_params, None, initial_opt_state, initial_rng_key)
        score_apply_fn = lambda params, data, rng_key, state_val: score_model.apply(params, rng_key, data)
    elif type(score_model) == hk.TransformedWithState:
        if score_state is None:
            initial_params, init_state = score_model.init(jrandom.PRNGKey(seed), 1.0*jnp.ones((2*vae_batch_size,dim*2+1)))
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
        dataset_epoch = vae_datasets.batch(vae_batch_size)
        for ds in dataset_epoch:
            ds = jnp.array(ds)
            score_state, score_loss = update_score(score_state, ds)
            vae_state, vae_loss = update_vae(vae_state, ds, training_type="Encoder")
            print(f"Epoch: {step+1} \t VAE Loss: {vae_loss:.4f} \t Score Loss: {score_loss:.4f}")
            save_model(score_path, score_state)
            save_model(vae_path, vae_state)
    for step in range(epochs_decoder):
        dataset_epoch = vae_datasets.batch(vae_batch_size)
        for ds in dataset_epoch:
            ds = jnp.array(ds)
            score_state, score_loss = update_score(score_state, ds)
            vae_state, vae_loss = update_vae(vae_state, ds, training_type="Decoder")
            print(f"Epoch: {step+1} \t VAE Loss: {vae_loss:.4f} \t Score Loss: {score_loss:.4f}")
            save_model(score_path, score_state)
            save_model(vae_path, vae_state)
    for step in range(epochs):
        dataset_epoch = vae_datasets.batch(vae_batch_size)
        for ds in dataset_epoch:
            ds = jnp.array(ds)
            score_state, score_loss = update_score(score_state, ds)
            vae_state, vae_loss = update_vae(vae_state, ds, training_type="All")
            save_model(score_path, score_state)
            save_model(vae_path, vae_state)
            print(f"Epoch: {step+1} \t VAE Loss: {vae_loss:.4f} \t Score Loss: {score_loss:.4f}")
            save_model(score_path, score_state)
            save_model(vae_path, vae_state)
    if (step+1) % save_step == 0:
        print(f"Epoch: {step+1} \t VAE Loss: {vae_loss:.4f} \t Score Loss: {score_loss:.4f}")
        save_model(score_path, score_state)
        save_model(vae_path, vae_state)
          
    save_model(score_path, score_state)
    save_model(vae_path, vae_state)
    
    return
