#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:51:27 2024

@author: fmry
"""

#%% Modules

#jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, lax, vmap, jacfwd, value_and_grad
from jax import tree_leaves, tree_map, tree_flatten, tree_unflatten #for saving model

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

#os
import os

#pickle
import pickle

#functools
from functools import partial

#data types
from jax import Array
from typing import Callable, Tuple, NamedTuple