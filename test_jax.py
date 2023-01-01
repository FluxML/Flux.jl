#%%
import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API

#import numpy as np                     # Ordinary NumPy
#import optax                           # Optimizers
#import tensorflow_datasets as tfds     # TFDS for MNIST
# %%
x = jnp.arange(16).reshape(1,2,2,4) / 16
y = nn.dot_product_attention(x, x, x)
yt = y.transpose((3,2,1,0))

yt
yt.shape
# %%
