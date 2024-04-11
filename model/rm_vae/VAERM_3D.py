#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:05:11 2023

@author: fmry
"""

#%% Sources

#%% Modules

#jax,
from jaxgeometry.setup import *
from jaxgeometry.manifolds import Latent

#jaxgeometry
from jaxgeometry.stochastics import product_sde, Brownian_coords

from jax.nn import elu, sigmoid, swish, tanh, softplus

#%% VAE Output

class VAEOutput(NamedTuple):
  z: Array
  mu_xz: Array
  log_sigma_xz: Array
  mu_zx: Array
  log_t_zx: Array
  mu_z: Array
  log_t_z: Array
  dW: Array
  dt: Array
  z_prior:Array
  dW_prior:Array
  dt_prior:Array
  G_mu_zx:Array
  G_mu_z:Array

#%% Other

@dataclasses.dataclass
class ScoreNet(hk.Module):
    
    dim:int
    layers:list
    
    def model(self)->object:
        
        model = []
        for l in self.layers:
            model.append(hk.Linear(l, w_init=jnp.zeros, b_init=jnp.zeros))
            model.append(tanh)
            
        model.append(hk.Linear(self.dim))
        
        return hk.Sequential(model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        
        x_new = x.T
        x1 = x_new[:self.dim].T
        x2 = x_new[self.dim:(2*self.dim)].T
        t = x_new[-1]
        
        shape = list(x.shape)
        shape[-1] = 1
        t = x_new[-1].reshape(shape)
            
        grad_euc = (x1-x2)/t
      
        return self.model()(x)+grad_euc

@dataclasses.dataclass
class Encoder(hk.Module):
        
    latent_dim : int = 2
    init:hk.initializers = hk.initializers.RandomNormal()
    
    def mu_layer(self, z:Array)->Array:

        return hk.Linear(output_size=self.latent_dim, w_init=self.init, b_init=self.init)(z)
    
    def log_t_layer(self, z:Array)->Array:
        
        return hk.Linear(output_size=1, w_init=jnp.zeros, b_init=jnp.zeros)(z)

    def __call__(self, x:Array) -> Tuple[Array, Array]:

        x = x.reshape(-1,3)
        
        x = swish(hk.Linear(output_size=100, b_init=self.init)(x))
        x = swish(hk.Linear(output_size=100, b_init=self.init)(x))

        mu_zx = self.mu_layer(x)
        log_t_zx = self.log_t_layer(x)

        return mu_zx, log_t_zx

@dataclasses.dataclass
class Decoder(hk.Module):
  """Decoder model."""
  init:hk.initializers = hk.initializers.RandomNormal()
  
  def mu_layer(self, z:Array)->Array:
      
      return hk.Linear(output_size=3, w_init=self.init, b_init=self.init)(z)
  
  def log_sigma_layer(self, z:Array)->Array:
      
      log_sigma = hk.Linear(output_size=1, w_init=jnp.zeros, b_init=jnp.zeros)(z)
      
      return jnp.tile(log_sigma, (1,3))

  def __call__(self, z: Array) -> Tuple[Array,Array]:
      
        z = swish(hk.Linear(output_size=100, w_init=self.init, b_init=self.init)(z))

        mu_xz = self.mu_layer(z)
        log_sigma_xz = self.log_sigma_layer(z)
        
        return mu_xz, log_sigma_xz
    
class PriorLayer(hk.Module):

  def __init__(self, output_size, name=None):
    super().__init__(name=name)
    self.output_size = output_size

  def __call__(self, x):
    #j, k = x.shape[-1], self.output_size
    #w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    #w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    b = hk.get_parameter("b", shape=[self.output_size], dtype=x.dtype, init=jnp.zeros)
    return b

class VAEBM(hk.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 seed:int=2712,
                 dt_steps:int=100,
                 sample_method:str='Local',
                 ):
        super(VAEBM, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.key = jrandom.key(seed)
        self.sample_method = sample_method
        self.dt_steps = dt_steps
        
    def muz(self, z:Array)->Array:
        
        mu_z = PriorLayer(output_size=z.shape[-1])(z)

        return mu_z*jnp.ones_like(z)
    
    def log_tz(self, z:Array)->Array:
        
        log_t_z = PriorLayer(output_size=1)(z)

        return log_t_z*jnp.ones((z.shape[0],1))
        
    def dts(self, T:float=1.0,n_steps:int=1000)->Array:
        """time increments, deterministic"""
        return jnp.array([T/n_steps]*n_steps)

    def dWs(self,d:int,_dts:Array=None,num:int=1)->Array:
        """
        standard noise realisations
        time increments, stochastic
        """
        keys = jrandom.split(self.key,num=num+1)
        self.key = keys[0]
        subkeys = keys[1:]
        if _dts == None:
            _dts = self.dts()
        if num == 1:
            return jnp.sqrt(_dts)[:,None]*jrandom.normal(subkeys[0],(_dts.shape[0],d))
        else:
            return vmap(lambda subkey: jnp.sqrt(_dts)[:,None]*jrandom.normal(subkey,(_dts.shape[0],d)))(subkeys) 
        
    def Jmu(self,z):

        return jacfwd(lambda z: self.decoder(z.reshape(-1,2))[0])(z)
    
    def Jsigma(self,z):

        return jacfwd(lambda z: self.decoder(z.reshape(-1,2))[1])(z)
        
    def G(self,z):

        Jmu = self.Jmu(z).squeeze()
        Jsigma = self.Jsigma(z).squeeze()
        
        return jnp.dot(Jmu.T,Jmu)+jnp.dot(Jsigma.T, Jsigma)
    
    def DG(self,z):
        
        return jacfwd(self.G)(z)
    
    def Ginv(self,z):
        
        return jnp.linalg.inv(self.G(z))
    
    def Chris(self,z):
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def euclidean_sample(self, mu:Array, log_t:Array):
        
        t = jnp.exp(log_t)
        eps = jrandom.normal(hk.next_rng_key(), mu.shape)
        dt = hk.vmap(lambda t: self.dts(t**2,self.dt_steps), split_rng=False)(t).squeeze().T
        
        return mu+t*eps, eps, dt[-1]
    
    def taylor_sample(self, mu:Array, log_t:Array):
        
        def sample(carry, step):
            
            t,z = carry
            dt, dW = step
            
            t += dt
            ginv = self.Ginv(z)
            
            stoch = jnp.dot(ginv, dW)
            z += stoch
            
            t = t.astype(carry[0].dtype)
            z = z.astype(carry[1].dtype)
            
            return ((t,z),)*2
        
        t = jnp.exp(2*log_t)
        dt = hk.vmap(lambda t: self.dts(t,self.dt_steps), split_rng=False)(t).squeeze().T
        N_data = mu.shape[0]
        dW = hk.vmap(lambda dt: self.dWs(self.encoder.latent_dim,dt),
                     split_rng=False)(dt).reshape(-1,N_data,self.encoder.latent_dim)
        
        #vmap(lambda mu,dt,dW: lax.scan(step, init=(mu,0.0), xs=(dt,dW)))(mu,dt,jnp.transpose(dW, axis=(1,0,2)))
        val, _ =hk.scan(lambda carry, step: hk.vmap(lambda t,z,dt,dW: sample((t,z),(dt,dW)),
                                                        split_rng=False)(carry[0],carry[1],step[0],step[1]),
                         init=(jnp.zeros_like(t),mu), xs=(dt,dW)
                         )

        return val[1], dW[-1], dt[-1]
    
    def local_sample(self, mu:Array, log_t:Array)->Array:
        
        def sample(carry, step):
            
            t,z = carry
            dt, dW = step
            
            t += dt
            ginv = self.Ginv(z)
            Chris = self.Chris(z)

            stoch = jnp.dot(ginv, dW)
            det = 0.5*jnp.einsum('jk,ijk->i', ginv, Chris)
            z += det+stoch
            
            t = t.astype(carry[0].dtype)
            z = z.astype(carry[1].dtype)
            
            return ((t,z),)*2
        
        t = jnp.exp(2*log_t)
        dt = hk.vmap(lambda t: self.dts(t,self.dt_steps), split_rng=False)(t).squeeze().T
        N_data = mu.shape[0]
        dW = hk.vmap(lambda dt: self.dWs(self.encoder.latent_dim,dt),
                     split_rng=False)(dt).reshape(-1,N_data,self.encoder.latent_dim)

        val, _ =hk.scan(lambda carry, step: hk.vmap(lambda t,z,dt,dW: sample((t,z),(dt,dW)), 
                                                        split_rng=False)(carry[0],carry[1],step[0],step[1]),
                         init=(jnp.zeros_like(t),mu), xs=(dt,dW)
                         )

        return val[1], dW[-1], dt[-1]

    def __call__(self, x: Array) -> VAEOutput:
        """Forward pass of the variational autoencoder."""
        x = x.astype(jnp.float32)
        mu_zx, log_t_zx = self.encoder(x)
        
        if self.sample_method == 'Local':
            z, dW, dt = self.local_sample(mu_zx, log_t_zx)
            mu_z, log_t_z = self.muz(z), self.log_tz(z)
            z_prior, dW_prior, dt_prior = self.local_sample(mu_z, log_t_z)
        elif self.sample_method == 'Taylor':
            z, dW, dt = self.taylor_sample(mu_zx, log_t_zx)
            mu_z, log_t_z = self.muz(z), self.log_tz(z)
            z_prior, dW_prior, dt_prior = self.taylor_sample(mu_z, log_t_z)
        elif self.sample_method == 'Euclidean':
            z, dW, dt = self.euclidean_sample(mu_zx, log_t_zx)
            mu_z, log_t_z = self.muz(z), self.log_tz(z)
            z_prior, dW_prior, dt_prior = self.euclidean_sample(mu_z, log_t_z)
        else:
            raise ValueError("Invalid sampling method. Choose either: Local, Taylor, Euclidean")
            
        mu_xz, log_sigma_xz = self.decoder(z)
        G_z = self.G(z)
        G_mu_zx = self.G(mu_zx)
        G_mu_z = self.G(mu_z)
        
        return VAEOutput(z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, dW, dt,
                         z_prior, dW_prior, dt_prior, G_mu_zx, G_mu_z)

#%% Transformed model
    
@hk.transform
def vae_model(x):
    
    vae = VAEBM(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae(x)

#%% Transformed Encoder model
    
@hk.transform
def model_encoder(x):
    
    vae = VAEBM(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae.encoder(x)[0]

#%% Transformed Decoder model
    
@hk.transform
def model_decoder(z):
    
    vae = VAEBM(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae.decoder(z)

@hk.transform
def score_model(x):
    
    score = ScoreNet(
    dim=2,
    layers=[50,100,200,200,100,50],
    )
  
    return score(x)
