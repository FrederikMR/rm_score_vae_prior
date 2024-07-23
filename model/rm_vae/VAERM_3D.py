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

#%% Score Net

@dataclasses.dataclass
class ScoreNet(hk.Module):
    
    dim:int
    layers:List
    
    def model(self)->object:
        
        model = []
        for l in self.layers:
            model.append(hk.Linear(l, w_init=jnp.zeros, b_init=jnp.zeros))
            model.append(tanh)
            
        model.append(hk.Linear(self.dim, w_init=jnp.zeros, b_init=jnp.zeros))
        
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

#%% Encoder

@dataclasses.dataclass
class Encoder(hk.Module):
        
    encoder_layers:List = [100, 100]
    mu_layers:List = []
    log_t_layers:List = []
    latent_dim : int = 2
    init:hk.initializers = hk.initializers.RandomNormal()
    
    def encoder_model(self)->object:
        
        model = []
        for l in self.encoder_layers:
            model.append(hk.Linear(l, w_init=self.init, b_init=self.init))
            model.append(swish)
        
        return hk.Sequential(model)
    
    def mu_model(self)->Array:
        
        model = []
        for l in self.mu_layers:
            model.append(hk.Linear(l, w_init=self.init, b_init=self.init))
            model.append(swish)
            
        model.append(hk.Linear(self.latent_dim, w_init=self.init, b_init=self.init))
        
        return hk.Sequential(model)
    
    def log_t_model(self)->Array:
        
        model = []
        for l in self.log_t_layers:
            model.append(hk.Linear(l, w_init=jnp.zeros, b_init=jnp.zeros))
            model.append(swish)
            
        model.append(hk.Linear(self.latent_dim, w_init=jnp.zeros, b_init=jnp.zeros))
        
        return hk.Sequential(model)

    def __call__(self, x:Array) -> Tuple[Array, Array]:

        x = x.reshape(-1,3)
        x_encoded = self.encoder_model()(x)

        mu_zx = self.mu_model()(x_encoded)
        log_t_zx = self.log_t_model()(x_encoded)

        return mu_zx, log_t_zx
    
#%% Decoder

@dataclasses.dataclass
class Decoder(hk.Module):
  """Decoder model."""
  decoder_layers:List = [100]
  mu_layers:List = []
  log_sigma_layers:List = []
  embedded_dim:int = 3
  init:hk.initializers = hk.initializers.RandomNormal()
  
  def decoder_model(self)->object:
      
      model = []
      for l in self.decoder_layers:
          model.append(hk.Linear(l, w_init=self.init, b_init=self.init))
          model.append(swish)
      
      return hk.Sequential(model)
  
  def mu_model(self)->Array:
      
      model = []
      for l in self.mu_layers:
          model.append(hk.Linear(l, w_init=self.init, b_init=self.init))
          model.append(swish)
          
      model.append(hk.Linear(self.embedded_dim, w_init=self.init, b_init=self.init))
      
      return hk.Sequential(model)
  
  def log_sigma_model(self)->Array:
      
      model = []
      for l in self.log_sigma_layers:
          model.append(hk.Linear(l, w_init=jnp.zeros, b_init=jnp.zeros))
          model.append(swish)
          
      model.append(hk.Linear(self.embedded_dim, w_init=jnp.zeros, b_init=jnp.zeros))
      
      return hk.Sequential(model)

  def __call__(self, z: Array) -> Tuple[Array,Array]:
      
        x_decoded = self.decoder_model()(z)

        mu_xz = self.mu_model()(x_decoded)
        log_sigma_xz = self.log_sigma_model()(x_decoded)
        
        return mu_xz, log_sigma_xz
    
#%% Prior Layer
    
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

#%% Riemannian Score Variational Prior

class VAERM(hk.Module):
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
        if sample_method not in ['Local', 'Taylor', 'Euclidean']:
            raise ValueError("Invalid sampling method. Choose either: Local, Taylor, Euclidean")
        else:
            self.sample_method = sample_method
        self.dt_steps = dt_steps
        
    def dts(self, T:float,n_steps:int)->Array:
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
        
    def JF(self,z:Array)->Array:
        
        return jacfwd(self.F)(z)
        
    def G(self,z:Array)->Array:
        
        JF = self.JF(z)
        
        return jnp.einsum('...ik,...il->...kl', JF, JF)
    
    def DG(self,z:Array)->Array:
        
        return jacfwd(self.G)(z)
    
    def Ginv(self,z:Array)->Array:
        
        return jnp.linalg.inv(self.G(z))
    
    def Chris(self,z:Array)->Array:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(jnp.einsum('...im,...kml->...ikl',gsharpx,Dgx)
                   +jnp.einsum('...im,...lmk->...ikl',gsharpx,Dgx)
                   -jnp.einsum('...im,...klm->...ikl',gsharpx,Dgx))
    
    def sample(self, mu:Array, log_t:Array):
        
        def euclidean_sample(mu:Array, log_t:Array):
            
            t = jnp.exp(log_t)
            eps = jrandom.normal(hk.next_rng_key(), mu.shape)
            dt = hk.vmap(lambda t: self.dts(t**2,self.dt_steps), split_rng=False)(t).squeeze().T
            
            return mu+t*eps, eps, dt[-1]
        
        def taylor_sample(z:Array, step:Tuple[Array, Array, Array])->Tuple[Array, Array]:

            dt, t, dW = step

            ginv = self.Ginv(z)
            z += jnp.einsum('...ik,...k->...i', ginv, dW)
            z = z.astype(self.dtype)
            
            return (z,)*2
        
        def local_sample(z:Array, step:Tuple[Array,Array,Array])->Tuple[Array,Array]:

            dt, t, dW = step

            ginv = self.Ginv(z)
            Chris = self.Chris(z)
            
            stoch = jnp.einsum('...ik,...k->...i', ginv, dW)
            det = 0.5*jnp.einsum('...jk,...ijk->...i', ginv, Chris)
            z += det+stoch
            z = z.astype(self.dtype)
            
            return (z,)*2

        if self.sample_method == 'Taylor':
            sample_step = taylor_sample
        elif self.sample_method == "Local":
            sample_step = local_sample
        elif self.sample_method == 'Euclidean':
            return euclidean_sample(mu, log_t)
            
        t = jnp.exp(2*log_t)
        dt = hk.vmap(lambda t: self.dts(t,self.dt_steps), split_rng=False)(t).squeeze().T
        t_grid = jnp.cumsum(dt, axis=-1)
        
        dW = hk.vmap(lambda dt: self.dWs(self.encoder.latent_dim,dt),
                     split_rng=False)(dt).reshape(self.dt_steps,-1,self.encoder.latent_dim)
        
        _, mut = hk.scan(sample_step, init=mu, xs=(dt, t_grid, dW))

        return mut, dW[-1], dt[-1]
    
    def muz(self, z:Array)->Array:
        
        mu_z = PriorLayer(output_size=z.shape[-1])(z)

        return mu_z*jnp.ones_like(z)
    
    def log_tz(self, z:Array)->Array:
        
        log_t_z = PriorLayer(output_size=1)(z)

        return log_t_z*jnp.ones((z.shape[0],1))

    def __call__(self, x: Array) -> VAEOutput:
        """Forward pass of the variational autoencoder."""
        mu_zx, log_t_zx = self.encoder(x)

        z, dW, dt = self.sample(mu_zx, log_t_zx)
        mu_z, log_t_z = self.muz(z), self.log_tz(z)
        z_prior, dW_prior, dt_prior = self.sample(mu_z, log_t_z)
            
        mu_xz, log_sigma_xz = self.decoder(z)
        
        return VAEOutput(z, mu_xz, log_sigma_xz, mu_zx, log_t_zx, mu_z, log_t_z, dW, dt,
                         z_prior, dW_prior, dt_prior)
    
#%% Riemannian Metric

class VAEG(hk.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 ):
        super(VAEBM, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        
    def JF(self,z:Array)->Array:
        
        return jacfwd(self.F)(z)
        
    def G(self,z:Array)->Array:
        
        JF = self.JF(z)
        
        return jnp.einsum('...ik,...il->...kl', JF, JF)
    
    def DG(self,z:Array)->Array:
        
        return jacfwd(self.G)(z)
    
    def Ginv(self,z:Array)->Array:
        
        return jnp.linalg.inv(self.G(z))
    
    def Chris(self,z:Array)->Array:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(jnp.einsum('...im,...kml->...ikl',gsharpx,Dgx)
                   +jnp.einsum('...im,...lmk->...ikl',gsharpx,Dgx)
                   -jnp.einsum('...im,...klm->...ikl',gsharpx,Dgx))

    def __call__(self, z: Array) -> VAEOutput:
        """Forward pass of the metric tensor."""
        
        return self.G(z)

#%% Transformed model
    
@hk.transform
def vae_model(x):
    
    vae = VAEBM(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return vae(x)

#%% Transformed metric
    
@hk.transform
def g_model(x):
    
    g = VAEG(
    encoder=Encoder(latent_dim=2),
    decoder=Decoder(),
    )
  
    return g(x)

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
