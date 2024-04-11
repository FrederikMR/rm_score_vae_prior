#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:05:11 2023

@author: fmry
"""

#%% Sources

#%% Modules

from rmvae.setup import *

#%% Euclidean ELBO

@partial(jit, static_argnames=['vae_model', 'training_type'])
def elbo_euclidean(x:Array,
                   vae_model:Callable[[Array], Array],
                   training_type:str="All"
                   )->Tuple[Array, [Array, Array]]:
    
    @jit
    def gaussian_likelihood(x:Array, mu_xz:Array, log_sigma_xz:Array)->Array:
        
        dim = mu_xz.shape[-1]
        diff = x-mu_xz
        sigma_xz_inv = jnp.exp(-2*log_sigma_xz)
        mu_term = jnp.sum(jnp.einsum('ij,ij->ij', diff**2, sigma_xz_inv), axis=-1)
        var_term = 2*dim*jnp.sum(log_sigma_xz, axis=-1)
        
        loss = -0.5*(mu_term+var_term)
        
        return jnp.mean(loss)
    
    @jit
    def kl_divergence(z:Array, mu_zx:Array, log_t_zx:Array, mu_z:Array, log_t_z:Array)->Array:
        
        dim = mu_zx.shape[-1]
        diff = z-mu_zx
        log_t_zx = log_t_zx.squeeze()
        log_t_z = log_t_z.squeeze()
        t_zx_inv = jnp.exp(-2*log_t_zx)
        t_z_inv = jnp.exp(-2*log_t_z)
        
        dist = jnp.sum(jnp.einsum('ij,i->ij', diff**2, t_zx_inv), axis=-1)
        log_qzx = dim*log_t_zx+dist
        
        diff = z-mu_z
        dist = jnp.sum(jnp.einsum('ij,i->ij', diff**2, t_z_inv), axis=-1)
        log_pz = dim*log_t_z+dist
        
        return 0.5*jnp.mean(log_pz-log_qzx)

    #z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, dW, dt, \
    #    z_prior, dW_prior, dt_prior, G_mu_zx, G_mu_z = vae_model(x)
    z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, *_ = vae_model(x)

    if training_type == "Encoder":
        log_sigma_xz = lax.stop_gradient(log_sigma_xz)
        mu_z = lax.stop_gradient(mu_z)
        log_t_z = lax.stop_gradient(log_t_z)
    elif training_type == "Decoder":
        mu_xz = lax.stop_gradient(mu_xz)
        log_t_zx = lax.stop_gradient(log_t_zx)
        mu_zx = lax.stop_gradient(mu_zx)

    rec_loss = gaussian_likelihood(x, mu_xz, log_sigma_xz)
    kld = kl_divergence(z, mu_zx, log_t_zx, mu_z, log_t_z)
    elbo = kld-rec_loss
    
    return elbo, (rec_loss, kld)

#%% Riemannian Distance ELBO

@partial(jit, static_argnames=['vae_model', 'training_type'])
def elbo_distance(x:Array,
                  vae_model:Callable[[Array], Array],
                  dist_fun:Callable[[Array, Array], Array],
                  training_type:str="All"
                  )->Tuple[Array, [Array, Array]]:
    
    @jit
    def gaussian_likelihood(x:Array, mu_xz:Array, log_sigma_xz:Array)->Array:
        
        dim = mu_xz.shape[-1]
        diff = x-mu_xz
        sigma_xz_inv = jnp.exp(-2*log_sigma_xz)
        mu_term = jnp.sum(jnp.einsum('ij,ij->ij', diff**2, sigma_xz_inv), axis=-1)
        var_term = 2*dim*jnp.sum(log_sigma_xz, axis=-1)
        
        loss = -0.5*(mu_term+var_term)
        
        return jnp.mean(loss)
    
    @jit
    def kl_divergence(z:Array, mu_zx:Array, log_t_zx:Array, mu_z:Array, log_t_z:Array,
                      G_mu_zx:Array, G_mu_z:Array)->Array:
        
        dim = mu_zx.shape[-1]

        logdet_G_mu_zx = jnp.linalg.slogdet(G_mu_zx).logabsdet
        logdet_G_mu_z = jnp.linalg.slogdet(G_mu_z).logabsdet
        
        dist_z_muzx = dist_fun(z, mu_zx)**2
        dist_z_muz = dist_fun(z, mu_z)**2
        
        log_qzx = dim*log_t_zx+dist
        
        log_qzx = dim*log_zx+dist_z_mu_zx-logdet_G_mu_zx
        log_pz = dim*log_t_z+dist_z_mu-logdet_G_mu_z
        
        return 0.5*jnp.mean(log_pz-log_qzx)

    z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, dW, dt, \
        z_prior, dW_prior, dt_prior, G_mu_zx, G_mu_z = vae_model(x)
    

    if training_type == "Encoder":
        log_sigma_xz = lax.stop_gradient(log_sigma_xz)
        mu_z = lax.stop_gradient(mu_z)
        log_t_z = lax.stop_gradient(log_t_z)
    elif training_type == "Decoder":
        mu_xz = lax.stop_gradient(mu_xz)
        log_t_zx = lax.stop_gradient(log_t_zx)
        mu_zx = lax.stop_gradient(mu_zx)

    rec_loss = gaussian_likelihood(x, mu_xz, log_sigma_xz)
    kld = kl_divergence(z, mu_zx, log_t_zx, mu_z, log_t_z, G_mu_zx, G_mu_z)
    elbo = kld-rec_loss
    
    return elbo, (rec_loss, kld)

#%% VAE Riemannian Fun

@partial(jit, static_argnames=['vae_model', 'score_model', 'training_type'])
def elbo_scores(x:Array,
                vae_model:Callable[[Array], Array],
                score_model:Callable[[Array], Array],
                training_type="All"
                )->Tuple[Array, [Array, Array]]:
    
    @jit
    def gaussian_likelihood(z:Array, mu_xz:Array, log_sigma_xz:Array)->Array:
        
        dim = mu_xz.shape[-1]
        diff = x-mu_xz
        sigma_xz_inv = jnp.exp(-2*log_sigma_xz)
        mu_term = jnp.sum(jnp.einsum('ij,ij->ij', diff**2, sigma_xz_inv), axis=-1)
        var_term = 2*dim*jnp.sum(log_sigma_xz, axis=-1)
        
        loss = -0.5*(mu_term+var_term)
        
        return loss
    
    @jit
    def kl_divergence(z:Array, s_logqzx:Array, s_logpz:Array)->Array:

        return jnp.einsum('...i,...i->...', s_logqzx-s_logpz, z)
    
    @jit
    def loss_fun(z:Array, mu_xz:Array, log_sigma_xz:Array, s_logqzx:Array, s_logpz:Array)->Array:
        
        rec = jnp.mean(gaussian_likelihood(z, mu_xz, log_sigma_xz))
        kld = jnp.mean(kl_divergence(z, s_logqzx, s_logpz))
        
        return rec-kld, rec, kld
            
    z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, *_ = vae_model(x)
    t_zx = jnp.exp(2*log_t_zx)
    t_z = jnp.exp(2*log_t_z)
    
    s_logqzx = lax.stop_gradient(score_model(jnp.hstack((mu_zx,z,t_zx.reshape(-1,1)))))
    s_logpz = lax.stop_gradient(score_model(jnp.hstack((mu_z, z, t_z.reshape(-1,1)))))
    
    if training_type == "Encoder":
        log_sigma_xz = lax.stop_gradient(log_sigma_xz)
        mu_z = lax.stop_gradient(mu_z)
        log_t_z = lax.stop_gradient(log_t_z)
    elif training_type == "Decoder":
        mu_xz = lax.stop_gradient(mu_xz)
        log_t_zx = lax.stop_gradient(log_t_zx)
        mu_zx = lax.stop_gradient(mu_zx)
    elbo, rec_loss, kld = loss_fun(z, mu_xz, log_sigma_xz, s_logqzx, s_logpz)

    return elbo, (rec_loss, kld)

#%% Denoising Score Matching First Order

@partial(jit, static_argnames=['score_model'])
def dsm(x0:Array,
        xt:Array,
        t:Array,
        dW:Array,
        dt:Array,
        score_model:Callable[[Array, Array, Array], Array],
        )->Array:

    s1 = score_model(jnp.hstack((x0,xt,t.reshape(-1,1))))
    
    return jnp.mean(jnp.sum(dW/dt+s1, axis=-1))
    
#%% Variance Reduction Denoising Score Matching First Order

@partial(jit, static_argnames=['score_model'])
def dsmvr(x0:Array,
          xt:Array,
          t:Array,
          dW:Array,
          dt:Array,
          score_model:Callable[[Array], Array]
          )->Array:
    
    s1 = score_model(jnp.hstack((x0,xt,t.reshape(-1,1))))
    s1p = score_model(jnp.shatck((x0,x0,t.reshape(-1,1))))
    
    z = dW/dt
    
    l1_loss = z+s1
    l1_loss = 0.5*jnp.einsum('...j,...j->...', l1_loss)
    var_loss = jnp.einsum('...j,...j->...', s1p, z)+jnp.einsum('...j,...j->...', z, z)
    
    return jnp.mean(l1_loss-var_loss)
    
    