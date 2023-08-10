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
        return [model(graph, key=skey)._replace(time=graph.time+1), key], graph

    return jax.lax.scan(_step, [graph, key], jnp.arange(steps))

def remove_isolates(graph: GGraph):

    nodes, edges, rec, send, anodes, aedges, *_ = graph
    n_nodes = nodes.shape[0]

    are_rec = jax.ops.segment_sum(jnp.ones((n_nodes,)), rec, n_nodes)
    arent_rec = jnp.where(are_rec>0., 0., 1.)

    are_send = jax.ops.segment_sum(jnp.ones((n_nodes,)), send, n_nodes)
    arent_semd = jnp.where(are_send>0., 0., 1.)

    are_iso = arent_rec * arent_send

    # remove isolated nodes
    nanodes = anodes * degens.astype(float)
    idxs = jnp.argsort(1.-nanodes)
    nanodes = nanodes[idxs]
    new_nodes = jnp.where(nanodes[:, None], nodes[idxs], 0.)

    # relabel edges with new indexes

    return graph._replace(nodes=nodes)

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

class Block(eqx.Module):

    cond_fn: t.Callable
    module: t.Callable

    def __init__(self, module:t.Callable, cond_fn: t.Callable):
        self.cond_fn = cond_fn
        self.module = module

    def __call__(self, graph: GGraph, key: jr.PRNGKey):
        return jax.lax.cond(
            self.cond_fn(graph, key),
            lambda g, k: self.module(g, k),
            lambda g, k: g,
            graph, key
        )

class Clamp(eqx.Module):

    idxs: list
    values: jax.Array

    def __init__(self, nodes_idx: t.Union[list, int], values: jax.Array):
        if isinstance(nodes_idx, list):
            assert len(nodes_idx) == values.shape[0]
        self.idxs = nodes_idx
        self.values = values

    def __call__(self, graph, key):
        return graph._replace(nodes=nodes.at[self.idxs].set(self.values))