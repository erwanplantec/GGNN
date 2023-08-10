from src.models._graph import GGraph
from src.models._utils import rollout

import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
import typing as t
from functools import partial


def ControlTask(statics: t.Callable, init_graph: GGraph, emb_fn: t.Callable, clamp_fn: t.Callable, target_fn: t.Callable):

	@partial(jax.vmap, in_axes=(None, 0))
	def _eval(key, params):
		key, eval_key, key_clamp, key_target = jr.split(key, 4)
		clamp = clamp_fn(key_clamp)
		params  = eqx.tree_at(lambda m: m.model.layers[0].values, clamp)
		target = target_fn(clamp, key_target)
		model = eqx.combine(params, statics)
		graph, graphs = model(init_graph, eval_key)
		bc = emb_fn(graphs)
		fit = jnp.mean(jnp.square(bc-target))
		return fit

	return _eval