#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:33:37 2023

@author: fmry
"""

#%% Sources

#%% Modules

from rmvae.setup import *

#%% VAE Sampling

class RMVAESampling(object):
    
    def __init__(self,
                 F:Callable[[Array],Array],
                 z0:Array,
                 sample_method:str='Local',
                 repeats:int=2**3,
                 z_samples:int=2**5,
                 t_samples:int=2**7,
                 max_T:float=1.0,
                 dt_steps:int=1000,
                 seed:int=2712
                 )->None:
        
        if z0.ndim == 1:
            self.z0s = jnp.tile(z0, (repeats,1))
        else:
            self.z0s = z0
        
        self.F = F
        self.sample_method = sample_method
        self.repeats = repeats
        self.z_samples = z_samples
        self.t_samples = t_samples
        self.N_sim = z_samples*repeats
        self.max_T = max_T
        self.dt_steps = dt_steps
        self.dtype = self.x0s.dtype
        
        self.dt = self.dts(T=max_T, n_steps=dt_steps)
        self.t_grid = jnp.cumsum(self.dt)
        self.key = jrandom.key(seed)
        
        return
        
    def __str__(self)->str:
        
        return "Generating Samples for Brownian Motion on Manifolds in Local Coordinates for VAE"
    
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
    
    def sample(self):
        
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

        dW = self.dWs(self.N_sim*self.dim,self.dt).reshape(-1,self.N_sim,self.dim)
        z0 = jnp.repeat(self.z0s, self.z_samples, axis=0)
        
        _, z = lax.scan(sample_step, init=z0, xs=(self.dt, self.t_grid, dW))

        return self.t_grid, z0, z, dW
        
    def __call__(self)->Tuple[Array, Array, Array, Array, Array]:
        
        while True:

            ts,z0s,z,dW = self.sample()
            self.z0s = z[-1,::self.z_samples]
            
            if jnp.isnan(jnp.sum(z)):
                self.z0s = self.z0s_default

            inds = jnp.array(random.sample(range(self.dt.shape[0]), self.t_samples))
            ts = ts[inds]
            samples = z[inds]
            
            yield jnp.hstack((jnp.tile(z0s,(self.t_samples,1)),
                              samples.reshape(-1,self.dim),
                              jnp.repeat(ts,self.N_sim).reshape((-1,1)),
                              dW[inds].reshape(-1,self.dim),
                              jnp.repeat(self.dt[inds],self.N_sim).reshape((-1,1)),
                              ))