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


def cosine_similarity(x, y):
    return jnp.dot(x, y) / (jnp.sqrt(jnp.sum(x**2)*jnp.sum(y**2))+1e-8)

def euclidean_dist(x, y):
    return jnp.sqrt(jnp.sum(jnp.square(x-y), axis=-1))

def jin(x, v):
    return jnp.where(x==v, True, False).any()


class Genesis(eqx.Module):

    """
    Actions:
        0: create new node
        1: create new edge
        2: do nothing
    policy_fn (callable) R^node_features -> R^3
    """
    #-------------------------------------------------------------------
    policy_fn: t.Callable
    query_fn: t.Callable
    dist_fn: t.Callable
    incr_nodes: t.Callable
    incr_edges: t.Callable
    sigma: float
    #-------------------------------------------------------------------

    def __init__(self, policy_fn: t.Callable, query_fn: t.Callable, max_nodes: int, 
                 max_edges: int, sigma: float = 0., dist_fn: t.Callable=cosine_similarity):

        self.policy_fn = policy_fn
        self.query_fn  = query_fn
        self.dist_fn = dist_fn
        self.sigma = sigma

        mat_n = incr_matrix(max_nodes)
        def incr_nodes(anodes, n):
            return jax.lax.fori_loop(
                0, n, lambda i, x: jnp.clip(x @ mat_n, 0., 1.), anodes
            ).at[-1].set(0.)
        self.incr_nodes = incr_nodes
        
        mat_e = incr_matrix(max_edges)
        def incr_edges(aedges, n):
            return jax.lax.fori_loop(
                0, n, lambda i, x: jnp.clip(x @ mat_e, 0., 1.), aedges
            ).at[-1].set(0.)
        self.incr_edges = incr_edges

    #-------------------------------------------------------------------

    def __call__(self, graph: GGraph, key: jr.PRNGKey, mode: str="soft"):

        pi_key, n_key, e_key = jr.split(key, 3)
        action_distribution = jax.vmap(self.policy_fn)(graph.nodes)
        if mode == "soft":
            actions = jr.categorical(pi_key, action_distribution, axis=-1)
        elif mode == "hard":
            actions = jnp.argmax(action_distribution, axis=-1)
        actions = jnp.where(graph.active_nodes, actions, 2)

        node_gens = jnp.where(actions==0, 1.0, 0.0)
        edge_gens = jnp.where(actions==1, 1.0, 0.0)

        graph = self.neurogenesis(graph, node_gens, n_key)
        graph = self.synaptogenesis(graph, edge_gens, e_key, mode=mode)

        return graph

    #-------------------------------------------------------------------

    def synaptogenesis(self, graph: GGraph, generators: jax.Array, 
                       key: jr.PRNGKey, mode: str="soft")->GGraph:
        
        key_edges, key_samp = jr.split(key)
        nodes, edges, rec, send, anodes, aedges, *_ = graph
        max_nodes, max_edges = nodes.shape[0], edges.shape[0]
        nids = jnp.arange(max_nodes)
        eids = jnp.arange(max_edges)
        n_active = anodes.sum().astype(int)
        e_active = aedges.sum().astype(int)
        
        # 1. Get dividing nodes
        gens = generators

        # 2. Select receivers
        queries = evmap(self.query_fn)(nodes)
        values = nodes
        scores = evmap(evmap(self.dist_fn, in_axes=(None, 0)), in_axes=(0, None))(queries, values)
        scores = jnp.clip(scores, -1e4, 1e4)
        scores = scores - (1.-anodes[None, :])*1e10
        scores = scores - jnp.identity(max_nodes)*1e10
        
        if mode == "soft":
            select = jnp.where(gens, jr.categorical(key_samp, scores, axis=-1).astype(int), 0) 
        elif mode == "hard":
            select = jnp.where(gens, jnp.argmax(scores, axis=-1).astype(int), 0)

        # 3. Check if edges do not already exist
        is_s = nids[:, None]==send[None, :]# (n, e)
        is_r = select[:, None]==rec[None, :]
        exist = jnp.logical_and(is_s, is_r).any(-1) & gens.astype(bool)
        gens = jnp.where(exist, 0., gens)
        
        # 4. Add new edges
        allowed = max_edges - e_active - 1
        n_gens = jnp.clip(gens.astype(int).sum(), 0, allowed)
        naedges = self.incr_edges(aedges, n_gens)
        mask_new_edges = naedges * (1-aedges)
        new_edges = edges + jr.normal(key_edges, edges.shape) * mask_new_edges[..., None]
        
        # 5. Add new senders
        trgets = jnp.cumsum(gens) * gens - gens
        trgets = jnp.where(gens, trgets.astype(int), -1) + e_active * gens.astype(int)
        nsend = jax.ops.segment_sum(nids, trgets, max_edges)
        nsend = (send * (1-mask_new_edges) + nsend).astype(int)
        

        # 6. Add receivers
        trgets = jnp.cumsum(gens) * gens - gens
        trgets = jnp.where(gens, trgets.astype(int), -1) + e_active * gens.astype(int)
        nrec = jax.ops.segment_sum(select, trgets, max_edges)
        nrec = (rec * (1-mask_new_edges) + nrec).astype(int)

        nrec = jnp.where(naedges, nrec, max_nodes-1)
        nsend = jnp.where(naedges, nsend, max_nodes-1)

        return graph._replace(edges        = new_edges, 
                              senders      = nsend,
                              receivers    = nrec, 
                              active_edges = naedges)

    #-------------------------------------------------------------------

    def neurogenesis(self, graph: GGraph, generators: jax.Array, 
                     key: jr.PRNGKey)->GGraph:

        key_div, key_nodes, key_edges = jr.split(key, 3)
        nodes, edges, rec, send, anodes, aedges, *_ = graph
        max_nodes, max_edges = nodes.shape[0], edges.shape[0]
        n_active = anodes.sum().astype(int)
        n_edges = aedges.sum().astype(int)
        ids = jnp.arange(max_nodes)
        eids = jnp.arange(max_edges)

        allowed = max_nodes - n_active - 1
        n_gens = jnp.clip(jnp.where(generators, 1, 0).sum(), 0, allowed)
        
        nanodes = self.incr_nodes(anodes, n_gens) #add new active nodes
        naedges = self.incr_edges(aedges, n_gens) #add new active edges
        
        mask_new_nodes = nanodes * (1-anodes)
        mask_new_edges = naedges * (1-aedges)
        
        trgets = jnp.cumsum(generators) * generators - generators
        trgets = jnp.where(generators, trgets.astype(int), -1) + n_edges * generators.astype(int)
        nsend = jax.ops.segment_sum(ids, trgets, max_edges)
        nsend = (send * (1-mask_new_edges) + nsend).astype(int)
        
        nrec = (jnp.cumsum(mask_new_edges)-1) * mask_new_edges + (n_active * mask_new_edges)
        nrec = (rec * (1-mask_new_edges) + nrec).astype(int)

        new_nodes = jax.ops.segment_sum(nodes, trgets, max_nodes)
        new_nodes = new_nodes + (jr.normal(key_nodes, nodes.shape) * mask_new_nodes[..., None] * self.sigma)
        new_edges = edges + jr.normal(key_edges, edges.shape) * mask_new_edges[..., None] 

        nrec = jnp.where(naedges, nrec, max_nodes-1)
        nsend = jnp.where(naedges, nsend, max_nodes-1)
        
        return graph._replace(nodes=new_nodes,
                              edges=new_edges,
                              active_nodes=nanodes,
                              active_edges=naedges,
                              receivers=nrec,
                              senders=nsend)

if __name__ == "__main__":
    gen = Genesis(lambda x: jnp.zeros((3,)), lambda x: x, 10, 10)