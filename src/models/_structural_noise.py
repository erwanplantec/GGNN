from src.models._graph import GGraph
from src.models._synaptogenesis import SynapticDegeneracy

import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import equinox.nn as nn


class SynapticNoise(eqx.Module):

	"""
	When called, remove some edges with probability <proba>
	Args:
		- proba (float): probability of doing smthing
		- strength (float): proba that an edge is removed
	"""
	#-------------------------------------------------------------------
	proba: float
	synaptic_deg: eqx.Module
	#-------------------------------------------------------------------

	def __init__(self, proba: float, strength: float, max_nodes: int, max_edges: int):
		
		self.proba = proba
		self.synaptic_deg = SynapticDegeneracy(lambda x: jnp.array([strength]), max_nodes, max_edges)

	#-------------------------------------------------------------------

	def __call__(self, graph: GGraph, key: jr.PRNGKey):

		key1, key2 = jr.split(key)
		return jax.lax.cond(jr.uniform(key1) < self.proba,
							lambda g, k: self.synaptic_deg(g, k),
							lambda g, k: g,
							graph, key2)
