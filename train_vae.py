#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:56:46 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
import jax.random as jrandom

import haiku as hk

import os

import tensorflow as tf

#argparse
import argparse

from jaxgeometry.statistics.vae.VAEBM3D import ScoreNet, Encoder, Decoder, VAEBM
from jaxgeometry.statistics.score_matching import train_vaebm, pretrain_vae, pretrain_scores, load_model

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data', default="Circle3D",
                        type=str)
    parser.add_argument('--data_path', default="data/vae/",
                        type=str)
    parser.add_argument('--score_loss_type', default="dsmvr",
                        type=str)
    parser.add_argument('--training_type', default="joint",
                        type=str)
    parser.add_argument('--sample_method', default="Local",
                        type=str)
    parser.add_argument('--vae_lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--score_lr_rate', default=0.0002,
                        type=float)
    parser.add_argument('--latent_dim', default=2,
                        type=int)
    parser.add_argument('--epochs', default=1000,
                        type=int)
    parser.add_argument('--vae_batch', default=100,
                        type=int)
    parser.add_argument('--use_pretrain_vae', default=0,
                        type=int)
    parser.add_argument('--use_pretrain_score', default=0,
                        type=int)
    parser.add_argument('--vae_split', default=0.0,#0.33,
                        type=float)
    parser.add_argument('--dt_steps', default=100,
                        type=int)
    parser.add_argument('--save_step', default=100,
                        type=int)
    parser.add_argument('--save_path', default='vaebm/joint_train/',
                        type=str)
    parser.add_argument('--vae_save_path', default='vaebm/pretrain_vae/',
                        type=str)
    parser.add_argument('--score_save_path', default='vaebm/pretrain_score/',
                        type=str)
    parser.add_argument('--seed', default=2712,
                        type=int)

    args = parser.parse_args()
    return args

#%% Code

def train():
    
    args = parse_args()
    
    if args.training_type == "vae":
        sample_method = "Euclidean"
    else:
        sample_method = args.sample_method
    sample_method = args.sample_method
    
    if args.data == "Circle3D":
        X = jnp.load(''.join((args.data_path, f'{args.data}.npy')))
        vae_datasets = tf.data.Dataset.from_tensor_slices(X).shuffle(buffer_size=10 * 100, seed=args.seed,
                                                                     reshuffle_each_iteration=True)
        
        @hk.transform
        def vae_model(x):
            
            
            vae = VAEBM(
            encoder=Encoder(latent_dim=args.latent_dim),
            decoder=Decoder(),
            sample_method = sample_method,
            dt_steps = args.dt_steps
            )
          
            return vae(x)
    
        @hk.transform
        def score_model(x):
            
            score = ScoreNet(
            dim=args.latent_dim,
            layers=[50,100,200,200,100,50],
            )
          
            return score(x)
        
        @hk.transform
        def decoder_model(z):
            
            decoder = Decoder()
          
            return decoder(z)
    
    vae_save_path = ''.join((args.vae_save_path, args.data, '/'))
    score_save_path = ''.join((args.score_save_path, args.data, '/'))
    save_path = ''.join((args.save_path, args.data, '/'))
    if not os.path.exists(vae_save_path):
        os.makedirs(vae_save_path)
    if not os.path.exists(score_save_path):
        os.makedirs(score_save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if not os.path.exists('vae/output/'):
        os.makedirs('vae/output/')
    if not os.path.exists('vae/error/'):
        os.makedirs('vae/error/')
        
    if args.training_type=="vae":
        pretrain_vae(vae_model=vae_model,
                     data_generator=vae_datasets,
                     lr_rate = args.vae_lr_rate,
                     save_path = vae_save_path,
                     split = args.vae_split,
                     batch_size=args.vae_batch,
                     vae_state = None,
                     epochs=args.epochs,
                     save_step = args.save_step,
                     vae_optimizer = None,
                     seed=args.seed,
                     )
    elif args.training_type == "score":
        
        if args.use_pretrain_vae:
            vae_state = load_model(vae_save_path)
        else:
            vae_state = None

        pretrain_scores(score_model=score_model,
                        vae_model = vae_model,
                        data_generator=vae_datasets,
                        vae_state=vae_state,
                        dim=args.latent_dim,
                        lr_rate = args.score_lr_rate,
                        batch_size = args.vae_batch,
                        save_path = score_save_path,
                        training_type=args.score_loss_type,
                        score_state = None,
                        epochs=args.epochs,
                        save_step=args.save_step,
                        score_optimizer = None,
                        seed=args.seed,
                        )
    elif args.training_type == "joint":
        vae_joint_save_path = ''.join((args.save_path, f'vae/{args.data}/'))
        score_joint_save_path = ''.join((args.save_path, f'scores/{args.data}/'))
        if not os.path.exists(vae_joint_save_path):
            os.makedirs(vae_joint_save_path)
        if not os.path.exists(score_joint_save_path):
            os.makedirs(score_joint_save_path)
        
        if args.use_pretrain_vae:
            vae_state = load_model(vae_save_path)
        else:
            vae_state = None
            
        if args.use_pretrain_score:
            score_state = load_model(score_save_path)
        else:
            score_state = None
        
        train_vaebm(vae_model=vae_model,
                    score_model=score_model,
                    vae_datasets=vae_datasets,
                    dim=args.latent_dim,
                    epochs=args.epochs,
                    vae_split=args.vae_split,
                    vae_optimizer = None,
                    score_optimizer = None,
                    vae_state=vae_state,
                    score_state=score_state,
                    seed=args.seed,
                    save_step=args.save_step,
                    score_type=args.score_loss_type,
                    vae_path=vae_joint_save_path,
                    score_path=score_joint_save_path,
                    )
    else:
        print("Invalid training type")
    
    return

#%% Main

if __name__ == '__main__':
        
    train()