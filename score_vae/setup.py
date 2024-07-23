#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:19:46 2024

@author: fmry
"""

#%% Import

#jax
from jax import Array
from jax import jit, lax, vmap, jacfwd, value_and_grad
from jax import tree_leaves, tree_flatten, tree_unflatten, tree_map

#jax.numpy
import jax.numpy as jnp

#numpy
import numpy as np

#jax.random
import jax.random as jrandom

#jax nn
from jax.nn import swish, tanh, gelu

#haiku
import haiku as hk

#tensorflow
import tensorflow as tf

#optax
import optax

#dataclasses
import dataclasses

#pickle
import pickle

#os
import os

#typing
from typing import Tuple, Callable, NamedTuple, List, Dict

#functools
from functools import partial

