#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:56:46 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp

import haiku as hk

import os

import tensorflow as tf

#argparse
import argparse

from score_vae.models.vae import VAE_MLP, Encoder_MLP, Decoder_MLP
from score_vae.models.score import Score_MLP
from score_vae.training import train_euclidean, train_score, train_score_vae
from score_vae.loader import load_model, load_tensor_slices

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data', default="Circle3D",
                        type=str)
    parser.add_argument('--data_path', default="data/",
                        type=str)
    parser.add_argument('--score_loss_type', default="dsm",
                        type=str)
    parser.add_argument('--training_type', default="joint",
                        type=str)
    parser.add_argument('--sample_method', default="euclidean",
                        type=str)
    parser.add_argument('--vae_lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--score_lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--latent_dim', default=2,
                        type=int)
    parser.add_argument('--embedded_dim', default=3,
                        type=int)
    parser.add_argument('--epochs', default=50000,
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--vae_split', default=0.0,#0.33,
                        type=float)
    parser.add_argument('--dt_steps', default=100,
                        type=int)
    parser.add_argument('--save_step', default=10,
                        type=int)
    parser.add_argument('--save_path', default='models/',
                        type=str)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% Code

def train():
    
    args = parse_args()
    
    save_path = ''.join((args.save_path, args.data, '/'))

    prevae_save_path = ''.join((save_path, f'pretrain_vae_{args.sample_method}/'))
    prescore_save_path = ''.join((save_path, f'pretrain_score_{args.sample_method}/'))
    
    vae_save_path = ''.join((save_path, f'vae_{args.sample_method}/'))
    score_save_path = ''.join((save_path, f'score_{args.sample_method}/'))
    
    if not os.path.exists(prevae_save_path):
        os.makedirs(prevae_save_path)
    if not os.path.exists(prescore_save_path):
        os.makedirs(prescore_save_path)
    if not os.path.exists(vae_save_path):
        os.makedirs(vae_save_path)
    if not os.path.exists(score_save_path):
        os.makedirs(score_save_path)
    
    if args.data == "Circle3D":
        X = jnp.load(''.join((args.data_path, args.data, '/data.npy')))
        data_generator = load_tensor_slices(X, batch_size=args.batch_size, seed=args.seed)
        
        @hk.transform
        def vae_model(x):
            
            
            vae = VAE_MLP(
            encoder=Encoder_MLP(latent_dim=args.latent_dim),
            decoder=Decoder_MLP(embedded_dim=args.embedded_dim),
            sample_method = args.sample_method,
            dt_steps = args.dt_steps,
            seed = args.seed,
            )
          
            return vae(x)
    
        @hk.transform
        def score_model(x):
            
            score = Score_MLP(
            dim=args.latent_dim,
            layers=[500,500,500,500,500],
            )
          
            return score(x)
        
    if args.training_type=="vae":
        train_euclidean(vae_model=vae_model,
                        data_generator=data_generator,
                        lr_rate = args.vae_lr_rate,
                        save_path = prevae_save_path,
                        split = args.vae_split,
                        batch_size=args.batch_size,
                        vae_state = None,
                        epochs=args.epochs,
                        save_step = args.save_step,
                        vae_optimizer = None,
                        seed=args.seed,
                        )
    elif args.training_type == "score":
        vae_state = load_model(prevae_save_path)

        train_score(score_model=score_model,
                        vae_model = vae_model,
                        vae_state = vae_state,
                        data_generator=data_generator,
                        dim=args.latent_dim,
                        lr_rate = args.score_lr_rate,
                        batch_size = args.batch_size,
                        save_path = prescore_save_path,
                        training_type=args.score_loss_type,
                        score_state = None,
                        epochs=args.epochs,
                        save_step=args.save_step,
                        score_optimizer = None,
                        seed=args.seed,
                        )
    elif args.training_type == "joint":
        vae_state = load_model(prevae_save_path)
        score_state = load_model(prescore_save_path)
        
        train_score_vae(vae_model=vae_model,
                    score_model=score_model,
                    data_generator=data_generator,
                    dim=args.latent_dim,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    vae_split=args.vae_split,
                    lr_rate_vae=args.score_lr_rate,
                    lr_rate_score=args.vae_lr_rate,
                    vae_optimizer = None,
                    score_optimizer = None,
                    vae_state=vae_state,
                    score_state=score_state,
                    seed=args.seed,
                    save_step=args.save_step,
                    score_type=args.score_loss_type,
                    vae_path=vae_save_path,
                    score_path=score_save_path,
                    )
    else:
        print("Invalid training type")
    
    return

#%% Main

if __name__ == '__main__':
        
    train()