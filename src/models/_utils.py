import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import PositionalSharding

import jraph
import equinox as eqx
import equinox.nn as nn
from equinox import filter_jit as ejit, filter_vmap as evmap, filter_pmap as epmap

from functools import partial
import typing as t

from src.models._graph import GGraph

cat = partial(jnp.concatenate, axis=-1)

def cat_args(f):
    def _f(*args):
        return f(cat([*args]))
    return _f

def incr_matrix(n):
    ns = jnp.arange(n)
    return jnp.zeros((n, n)).at[ns, ns].set(1.).at[ns, jnp.mod(ns+1, n)].set(1.)

def pow_dot(m, n:int):
    return jax.lax.fori_loop(0, n-1, lambda i, x: x@m, m)

def off_sigmoid(x, off=3.):
    """offsetted sigmoid function"""
    return jax.nn.sigmoid(x-off)

def rollout(model: t.Callable, graph: GGraph, key: jr.PRNGKey, steps: int):

    def _step(carry, x):
        graph, key = carry
        key, skey = jr.split(key)
        return [model(graph, key=skey), key], graph

    return jax.lax.scan(_step, [graph, key], jnp.arange(steps))


def shard_ggraph(graph: GGraph, sharding: PositionalSharding):
    nodes = jax.device_put(graph.nodes, sharding)
    edges = jax.device_put(graph.edges, sharding)
    receivers = jax.device_put(graph.receivers, sharding.reshape((-1, )))
    senders = jax.device_put(graph.senders, sharding.reshape((-1, )))
    active_nodes = jax.device_put(graph.active_nodes, sharding.reshape((-1,)))
    active_edges = jax.device_put(graph.active_edges, sharding.reshape((-1,)))
    
    return graph._replace(nodes        = nodes, 
                          edges        = edges, 
                          receivers    = receivers,
                          senders      = senders, 
                          active_nodes = active_nodes,
                          active_edges = active_edges)

def get_active_graph(graph: GGraph):
    """NOT JITTABLE"""
    node_mask = graph.active_nodes.astype(bool)
    edge_mask = graph.active_edges.astype(bool)

    return GGraph._replace(nodes        = graph.nodes[node_mask],
                           edges        = graph.edges[edge_mask],
                           n_node       = node_mask.astype(int).sum(),
                           n_edge       = edge_mask.astype(int).sum(),
                           receivers    = graph.receivers[edge_mask],
                           senders      = graph.senders[edge_mask],
                           active_edges = graph.active_edges[edge_mask],
                           active_nodes = graph.active_nodes[node_mask])