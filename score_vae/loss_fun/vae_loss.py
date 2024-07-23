#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:23:39 2024

@author: fmry
"""

#%% Sources

#%% Modules

from score_vae.setup import *
from score_vae.models.vae import VAEOutput

#%% Euclidean ELBO

@partial(jit, static_argnames=['vae_output', 'training_type'])
def elbo_euclidean(x:Array,
                   vae_output:VAEOutput,
                   training_type:str="All"
                   )->Tuple[Array, Tuple[Array, Array]]:
    
    @jit
    def gaussian_likelihood(x:Array, mu_xz:Array, log_sigma_xz:Array)->Array:

        diff = x-mu_xz
        sigma_xz_inv = jnp.exp(-2*log_sigma_xz)
        mu_term = jnp.sum(jnp.einsum('ij,ij->ij', diff**2, sigma_xz_inv), axis=-1)
        var_term = 2.0*jnp.sum(log_sigma_xz, axis=-1)
        
        return -0.5*jnp.mean(mu_term+var_term)
    
    @jit
    def kl_divergence(z:Array, mu_zx:Array, log_t_zx:Array, mu_z:Array, log_t_z:Array)->Array:
        
        dim = mu_zx.shape[-1]
        
        log_t_zx = log_t_zx.squeeze()
        log_t_z = log_t_z.squeeze()
        t_zx_inv = jnp.exp(-log_t_zx)
        t_z_inv = jnp.exp(-log_t_z)
        
        diff = z-mu_zx
        
        dist = jnp.sum(jnp.einsum('ij,i->ij', diff**2, t_zx_inv), axis=-1)
        log_qzx = dim*log_t_zx+dist
        
        diff = z-mu_z
        dist = jnp.sum(jnp.einsum('ij,i->ij', diff**2, t_z_inv), axis=-1)
        log_pz = dim*log_t_z+dist
        
        return -0.5*jnp.mean(log_qzx-log_pz)
    
    x, mu_xz, log_sigma_xz = vae_output.x, vae_output.mu_xz, vae_output.log_sigma_xz #Decoder
    z, mu_zx, log_t_zx, mu_z, log_t_z = vae_output.z, vae_output.mu_zx, vae_output.log_t_zx, vae_output.mu_z, vae_output.log_t_z #encoder

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

@partial(jit, static_argnames=['vae_output', 'training_type'])
def elbo_distance(x:Array,
                  vae_output:VAEOutput,
                  training_type:str="All",
                  )->Tuple[Array, Tuple[Array, Array]]:
    
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
        
        t_zx = jnp.exp(log_t_zx)
        t_z = jnp.exp(log_t_z)
        
        log_qzx = dim*log_t_zx+dist_z_mu_zx2/t_zx+logdet_G_mu_zx #scaled with -2
        log_pz = dim*log_t_z+dist_z_mu_z2/t_z+logdet_G_mu_z #scaled with -2
        
        return -0.5*jnp.mean(log_qzx-log_pz)

    x, mu_xz, log_sigma_xz = vae_output.x, vae_output.mu_xz, vae_output.log_sigma_xz #Decoder
    z, mu_zx, log_t_zx, mu_z, log_t_z = vae_output.z, vae_output.mu_zx, \
        vae_output.log_t_zx, vae_output.mu_z, vae_output.log_t_z #encoder
    G_mu_zx, G_mu_z = vae_output.G_mu_zx, vae_output.G_mu_z
    dist_z_mu_zx2, dist_z_mu_z2 = vae_output.dist_z_mu_zx**2, vae_output.dist_z_mu_z**2

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

@partial(jit, static_argnames=['vae_output', 'score_model', 'training_type'])
def elbo_scores(x:Array,
                vae_output:VAEOutput,
                s_log_qzx:Array,
                s_log_pz:Array,
                training_type="All"
                )->Tuple[Array, Tuple[Array, Array]]:
    
    @jit
    def gaussian_likelihood(z:Array, mu_xz:Array, log_sigma_xz:Array)->Array:
        
        dim = mu_xz.shape[-1]
        diff = x-mu_xz
        sigma_xz_inv = jnp.exp(-2*log_sigma_xz)
        mu_term = jnp.sum(jnp.einsum('ij,ij->ij', diff**2, sigma_xz_inv), axis=-1)
        var_term = 2*dim*jnp.sum(log_sigma_xz, axis=-1)
        
        loss = -0.5*(mu_term+var_term)
        
        return jnp.mean(loss)
    
    @jit
    def kl_divergence(z:Array, s_log_qzx:Array, s_log_pz:Array)->Array:

        return jnp.mean(jnp.einsum('...i,...i->...', s_log_qzx-s_log_pz, z))
    
    @jit
    def loss_fun(z:Array, mu_xz:Array, log_sigma_xz:Array, s_log_qzx:Array, s_log_pz:Array)->Array:
        
        rec = jnp.mean(gaussian_likelihood(z, mu_xz, log_sigma_xz))
        kld = jnp.mean(kl_divergence(z, s_log_qzx, s_log_pz))
        
        return kld-rec, rec, kld
            
    x, mu_xz, log_sigma_xz = vae_output.x, vae_output.mu_xz, vae_output.log_sigma_xz #Decoder
    z, mu_zx, log_t_zx, mu_z, log_t_z = vae_output.z, vae_output.mu_zx, \
        vae_output.log_t_zx, vae_output.mu_z, vae_output.log_t_z #encoders
    
    s_log_qzx = lax.stop_gradient(s_log_qzx)
    s_log_pz = lax.stop_gradient(s_log_pz)
    if training_type == "Encoder":
        log_sigma_xz = lax.stop_gradient(log_sigma_xz)
        mu_z = lax.stop_gradient(mu_z)
        log_t_z = lax.stop_gradient(log_t_z)
    elif training_type == "Decoder":
        mu_xz = lax.stop_gradient(mu_xz)
        log_t_zx = lax.stop_gradient(log_t_zx)
        mu_zx = lax.stop_gradient(mu_zx)
        
    rec_loss = gaussian_likelihood(x, mu_xz, log_sigma_xz)
    kld = kl_divergence(z, s_log_qzx, s_log_pz)
    elbo = kld-rec_loss

    return elbo, (rec_loss, kld)