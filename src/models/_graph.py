import jax
import jax.numpy as jnp
import jax.random as jr

import jraph
import equinox as eqx
import equinox.nn as nn
from equinox import filter_jit as ejit, filter_vmap as evmap, filter_pmap as epmap

from functools import partial
import typing as t


class GGraph(t.NamedTuple):
    nodes: jnp.array
    edges: jnp.array
    receivers: jnp.array
    senders: jnp.array
    active_nodes: jnp.array #mask of active nodes (1. if active)
    active_edges: jnp.array #mask of active edges (1. if active)
    n_node: int = 1
    n_edge: int = 1
    time: int = 0
    globals: jnp.array = None

