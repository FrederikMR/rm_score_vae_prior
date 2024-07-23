#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:05:11 2023

@author: fmry
"""

#%% Sources

#%% Modules

from score_vae.setup import *
#from score_vae.models.vae import VAEOutput

#%% VAE Output

class VAEOutput(NamedTuple):
  x: Array
  mu_xz:Array
  log_sigma_xz:Array
  z:Array
  z_prior:Array
  mu_zx:Array
  log_t_zx:Array
  mu_z:Array
  log_t_z:Array
  dW_zx:Array
  dW_z:Array
  dt_zx:Array
  dt_z:Array

#%% Encoder

class Encoder_MLP(hk.Module):
    def __init__(self,
                 encoder_layers:List = [100, 100],
                 encoder_act:List = [gelu, gelu],
                 mu_layers:List = [],
                 mu_act:List = [],
                 log_t_layers:List = [],
                 log_t_act:List = [],
                 latent_dim : int = 2,
                 init:hk.initializers = None,
                 )->None:
        super().__init__()
        
        self.encoder_layers = encoder_layers
        self.encoder_act = encoder_act
        self.mu_layers = mu_layers
        self.mu_act = mu_act
        self.log_t_layers = log_t_layers
        self.log_t_act = log_t_act
        self.latent_dim = latent_dim
        self.init = init
        
        return
    
    def encoder_model(self)->object:
        
        model = []
        for layer,act in zip(self.encoder_layers,self.encoder_act):
            model.append(hk.Linear(layer))#, w_init=self.init, b_init=self.init))
            model.append(act)
        
        return hk.Sequential(model)
    
    def mu_model(self)->Array:
        
        model = []
        for layer,act in zip(self.mu_layers, self.mu_act):
            model.append(hk.Linear(layer))#, w_init=self.init, b_init=self.init))
            model.append(act)
            
        model.append(hk.Linear(self.latent_dim))#, w_init=self.init, b_init=self.init))
        
        return hk.Sequential(model)
    
    def log_t_model(self)->Array:
        
        model = []
        for layer,act in zip(self.log_t_layers, self.log_t_act):
            model.append(hk.Linear(layer, w_init=jnp.zeros, b_init=jnp.zeros))
            model.append(act)
            
        model.append(hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros))
        
        return hk.Sequential(model)

    def __call__(self, x:Array) -> Tuple[Array, Array]:

        x = x.reshape(-1,x.shape[-1])
        x_encoded = self.encoder_model()(x)

        mu_zx = self.mu_model()(x_encoded)
        log_t_zx = self.log_t_model()(x_encoded).squeeze()

        return mu_zx, log_t_zx
    
#%% Decoder

class Decoder_MLP(hk.Module):
    def __init__(self,
                 embedded_dim:int = 3,
                 decoder_layers:List = [100],
                 decoder_act:List = [gelu],
                 mu_layers:List = [],
                 mu_act:List = [],
                 log_sigma_layers:List = [],
                 log_sigma_act:List = [],
                 init:hk.initializers = hk.initializers.RandomNormal(),
                 )->None:
        super().__init__()
        
        self.embedded_dim = embedded_dim
        self.decoder_layers = decoder_layers
        self.decoder_act = decoder_act
        self.mu_layers = mu_layers
        self.mu_act = mu_act
        self.log_sigma_layers =log_sigma_layers
        self.log_sigma_act = log_sigma_act
        self.init = init
        
        return

    def decoder_model(self)->object:
        
        model = []
        for layer,act in zip(self.decoder_layers,self.decoder_act):
            model.append(hk.Linear(layer))#, w_init=self.init, b_init=self.init))
            model.append(act)
        
        return hk.Sequential(model)
  
    def mu_model(self)->Array:
        
        model = []
        for layer,act in zip(self.mu_layers,self.mu_act):
            model.append(hk.Linear(layer))#, w_init=self.init, b_init=self.init))
            model.append(act)
            
        model.append(hk.Linear(self.embedded_dim))#, w_init=self.init, b_init=self.init))
        
        return hk.Sequential(model)
  
    def log_sigma_model(self)->Array:
        
        model = []
        for layer,act in zip(self.log_sigma_layers,self.log_sigma_act):
            model.append(hk.Linear(layer, w_init=jnp.zeros, b_init=jnp.zeros))
            model.append(act)
            
        model.append(hk.Linear(self.embedded_dim, w_init=jnp.zeros, b_init=jnp.zeros))
        
        return hk.Sequential(model)

    def __call__(self, z: Array) -> Tuple[Array,Array]:
        
          x_decoded = self.decoder_model()(z)
    
          mu_xz = self.mu_model()(x_decoded)
          log_sigma_xz = self.log_sigma_model()(x_decoded)
          
          return mu_xz, log_sigma_xz
    
#%% Prior Layer
    
class PriorLayer(hk.Module):

  def __init__(self, output_size, dtype, name=None):
    super().__init__(name=name)
    
    self.output_size = output_size
    self.dtype = dtype
    
  def __call__(self):
    #j, k = x.shape[-1], self.output_size
    #w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    #w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    b = hk.get_parameter("b", shape=[self.output_size], dtype=self.dtype, init=jnp.zeros)
    
    return b

#%% Riemannian Score Variational Prior

class VAE_MLP(hk.Module):
    def __init__(self,
                 encoder:Encoder_MLP,
                 decoder:Decoder_MLP,
                 seed:int=2712,
                 dt_steps:int=100,
                 sample_method:str='Local',
                 ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.key = jrandom.key(seed)
        if sample_method not in ['local', 'taylor', 'euclidean']:
            raise ValueError("Invalid sampling method. Choose either: local, taylor, euclidean")
        else:
            self.sample_method = sample_method
            
        if self.sample_method == 'taylor':
            self.sample_step = self.taylor_sample
        elif self.sample_method == "local":
            self.sample_step = self.local_sample
        elif self.sample_method == 'euclidean':
            self.sample_step = self.euclidean_sample
            
        self.dt_steps = dt_steps
        
        self.key = jrandom.PRNGKey(seed)
        self.seed = seed
        
        return
        
    def dt(self,
           T:float
           )->Array:
        
        return T/self.dt_steps
        
    def dts(self, 
            T:float
            )->Array:
        
        return jnp.array([T/self.dt_steps]*self.dt_steps)

    def dWs(self,
            d:int,
            dts:Array=None,
            )->Array:
        """
        standard noise realisations
        time increments, stochastic
        """
        if dts == None:
            dts = self.dts()

        return jnp.sqrt(dts)[:,None]*jrandom.normal(hk.next_rng_key(),(dts.shape[0],d))
        
    def JF_mu(self,z:Array)->Array:
        
        return jacfwd(lambda z: self.decoder(z)[0])(z)
    
    def JF_sigma(self,z:Array)->Array:
        
        return jacfwd(lambda z: jnp.exp(self.decoder(z)[1]))(z)
        
    def G(self,z:Array)->Array:
        
        JF_mu = self.JF_mu(z)
        JF_sigma = self.JF_sigma(z)
        
        return jnp.einsum('...ik,...il->...kl', JF_mu, JF_mu)+\
            jnp.einsum('...ik,...il->...kl', JF_sigma, JF_sigma)
    
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
    
    def euclidean_sample(self,
                         carry:Tuple[Array,Array], 
                         step:Tuple[Array, Array, Array]
                         )->Tuple[Array, Array]:
        
        t, z = carry
        dt, dW = step
        
        t += dt
        z += dW
        z = z.astype(z.dtype)

        return ((t,z),)*2
    
    def taylor_sample(self,
                      carry:Tuple[Array,Array], 
                      step:Tuple[Array, Array, Array],
                      )->Tuple[Array, Array]:

        t, z = carry
        dt, dW = step

        t += dt

        ginv = vmap(self.Ginv)(z)
        z += jnp.einsum('...ik,...k->...i', ginv, dW)
        z = z.astype(z.dtype)
        
        return ((t,z),)*2
    
    def local_sample(self,
                     carry:Tuple[Array,Array], 
                     step:Tuple[Array,Array,Array]
                     )->Tuple[Array,Array]:

        t, z = carry
        dt, dW = step

        t += dt

        ginv = vmap(self.Ginv)(z)
        Chris = vmap(self.Chris)(z)
        
        stoch = jnp.einsum('...ik,...k->...i', ginv, dW)
        det = (0.5*jnp.einsum('...jk,...ijk->...i', ginv, Chris)*dt.reshape(-1,1)).squeeze()
        z += det+stoch
        z = z.astype(z.dtype)
        
        return ((t,z),)*2
    
    def sample(self,
               mu:Array,
               log_t:Array,
               )->Tuple[Array,Array,Array]:
        
        t = jnp.exp(log_t)
        dt = hk.vmap(lambda t: self.dts(t), split_rng=False)(t).T
        
        dW = jnp.einsum('ij,ijk->ijk', jnp.sqrt(dt),
                        jrandom.normal(hk.next_rng_key(),(dt.shape[0],
                                                          dt.shape[1],
                                                          self.encoder.latent_dim))).reshape(self.dt_steps,
                                                                                             -1,
                                                                                             self.encoder.latent_dim
                                                                                             )
        
        _, out = hk.scan(self.sample_step, 
                         init=(jnp.zeros(len(t)),mu), 
                         xs=(dt, dW))
        
        t,mut = out[0], out[1]

        return mut[-1], dW[-1], dt[-1]
    
    def prior(self, 
              z:Array
              )->Tuple[Array,Array]:
        
        mu_z = PriorLayer(output_size=z.shape[-1], dtype=z.dtype)()
        log_t_z = PriorLayer(output_size=1, dtype=z.dtype)()

        return mu_z*jnp.ones(z.shape), log_t_z*jnp.ones((z.shape[0]))

    def __call__(self, 
                 x:Array, 
                 ) -> VAEOutput:
        """Forward pass of the variational autoencoder."""
        mu_zx, log_t_zx = self.encoder(x)
        z, dW_zx, dt_zx = self.sample(mu_zx, log_t_zx)
        mu_z, log_t_z = self.prior(z)
        z_prior, dW_z, dt_z = self.sample(mu_z, log_t_z)
        
        mu_xz, log_sigma_xz = self.decoder(z)
        
        return VAEOutput(x,
                         mu_xz,
                         log_sigma_xz,
                         z,
                         z_prior,
                         mu_zx,
                         log_t_zx,
                         mu_z,
                         log_t_z,
                         dW_zx,
                         dW_z,
                         dt_zx,
                         dt_z,
                         )