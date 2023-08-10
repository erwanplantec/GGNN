from src.models._utils import *
from src.models._graph import GGraph

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from equinox import filter_jit as ejit, filter_vmap as evmap, filter_pmap as epmap

from functools import partial
import typing as t


class Clamping(eqx.Module):

	#-------------------------------------------------------------------
	mask: jax.Array
	#-------------------------------------------------------------------

	def __init__(self, mask: jax.Array):
		
		self.mask = mask.astype(int)

	#-------------------------------------------------------------------

	def __call(self, graph: GGraph, key: jr.PRNGKey):

		clamp_val = graph.holder["clamp"]

		return graph._replace(nodes = jnp.where(self.mask, clamp_val, graph.nodes))

	#-------------------------------------------------------------------