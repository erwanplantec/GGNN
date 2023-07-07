import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from equinox import filter_jit as ejit, filter_vmap as evmap, filter_pmap as epmap

from functools import partial
import typing as t

from src.models._utils import *
from src.models._graph import GGraph

#=================================================================================================
#=================================================================================================
#=================================================================================================

class SynaptoGenesis(eqx.Module):

    """
    Edge adding layer
        1. Compute edge generation probability pi for each node i
            pi = prob_fn(hi)
        2. Sample generating nodes
        3. For each of these compute a query
            qi = query_fn(hi)
        4. Get score sij for each pair ij wrt qi and hi
            sij = hj @ qi
        5. Sample a target j* each generating node i with:
            j* ~ softmax(sij) if mode is "soft"
            or
            j* = argmax(sij) if mode is "hard"
        6. Add a new edge eij* for each generating node i with random embedding
            
    """
    #-------------------------------------------------------------------
    query_fn: t.Callable    
    prob_fn: t.Callable
    max_nodes: int
    max_edges: int
    eincr_fn: t.Callable
    #-------------------------------------------------------------------

    def __init__(self, prob_fn, query_fn, max_nodes, max_edges):
        
        self.query_fn = query_fn
        self.prob_fn = prob_fn
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        mat_e = incr_matrix(max_edges)
        def incr_edges(aedges, n):
            return jax.lax.fori_loop(
                0, n, lambda i, x: jnp.clip(x @ mat_e, 0., 1.), aedges
            ).at[-1].set(0.)
        self.eincr_fn = incr_edges
        
    #-------------------------------------------------------------------

    @ejit
    def __call__(self, graph: GGraph, key: jr.PRNGKey, mode: str = "soft"):
        
        key_prob, key_edges, key_samp = jr.split(key, 3)
        nodes, edges, rec, send, _, _, _, anodes, aedges = graph
        nids = jnp.arange(self.max_nodes)
        eids = jnp.arange(self.max_edges)
        n_active = anodes.sum().astype(int)
        e_active = aedges.sum().astype(int)
        
        probs = jax.vmap(self.prob_fn)(nodes)[..., 0]
        gens = jr.uniform(key_prob, (self.max_nodes,)) < probs * anodes
        gens = gens.astype(float)
        
        allowed = self.max_edges - e_active - 1
        n_gens = jnp.clip(gens.astype(int).sum(), 0, allowed)
        naedges = self.eincr_fn(aedges, n_gens)
        mask_new_edges = naedges * (1-aedges)
        
        new_edges = edges + jr.normal(key_edges, edges.shape) * mask_new_edges[..., None]
        
        trgets = jnp.cumsum(gens) * gens - gens
        trgets = jnp.where(gens, trgets.astype(int), -1) + e_active * gens.astype(int)
        nsend = jax.ops.segment_sum(nids, trgets, self.max_edges)
        nsend = (send * (1-mask_new_edges) + nsend).astype(int)
        
        queries = evmap(self.query_fn)(nodes)
        scores = evmap(evmap(jnp.dot, in_axes=(None, 0)), in_axes=(0, None))(queries, nodes)
        scores = jnp.clip(scores, -1e4, 1e4)
        scores = scores - (1.-anodes[None, :])*1e10

        if mode == "soft":
            select = jnp.where(gens, jr.categorical(key_samp, scores, axis=-1).astype(int), 0) 
        elif mode == "hard":
            select = jnp.where(gens, jnp.argmax(scores, axis=-1).astype(int), 0)
        trgets = jnp.cumsum(gens) * gens - gens
        trgets = jnp.where(gens, trgets.astype(int), -1) + e_active * gens.astype(int)
        nrec = jax.ops.segment_sum(select, trgets, self.max_edges)
        nrec = (rec * (1-mask_new_edges) + nrec).astype(int)

        return graph._replace(edges        = new_edges, 
                              senders      = nsend,
                              receivers    = nrec, 
                              active_edges = naedges)


#=================================================================================================
#=================================================================================================
#=================================================================================================

class SynapticDegeneracy(eqx.Module):

    """
    Edge pruning layer
        1. Compute removal probability for each edge
            pij = prob_fn(eij)
        2. Remove edge ij with probabiliti pij
    """
    #-------------------------------------------------------------------
    prob_fn: t.Callable
    max_nodes: int
    max_edges: int
    #-------------------------------------------------------------------

    def __init__(self, prob_fn: t.Callable, max_nodes: int, max_edges: int):
        
        self.prob_fn = prob_fn
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
    #-------------------------------------------------------------------

    @ejit
    def __call__(self, graph: GGraph, key: jr.PRNGKey):
        
        key, key_prob = jr.split(key)
        nodes, edges, rec, send, _, _, _, anodes, aedges = graph

        probs = jax.vmap(self.prob_fn)(edges)[:,0] * aedges
        degens = jr.uniform(key_prob, (self.max_edges,)) < probs
        degens = degens.astype(float)
        naedges = aedges * (1.-degens)
        idxs = jnp.argsort(1.-naedges) #actives first, other after

        naedges = naedges[idxs] #actives first, other after
        nrec = rec[idxs]
        nrec = jnp.where(naedges, nrec, self.max_nodes-1)
        nsend = send[idxs]
        nsend = jnp.where(naedges, nsend, self.max_nodes-1)
        new_edges = jnp.where(naedges[:, None], edges[idxs], 0.)

        return graph._replace(active_edges = naedges,
                              senders      = nsend,
                              receivers    = nrec,
                              edges        = new_edges)










