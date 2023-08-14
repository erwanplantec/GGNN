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

def cosine_similarity(x, y):
    return jnp.dot(x, y) / (jnp.sqrt(jnp.sum(x**2)*jnp.sum(y**2))+1e-8)

def euclidean_dist(x, y):
    return jnp.sqrt(jnp.sum(jnp.square(x-y), axis=-1))

def jin(x, v):
    return jnp.where(x==v, True, False).any()

class QueryKey(eqx.Module):

    """
    Simple query and key generator
        Q = X @ W_Q
        K = X @ W_K
    where X has input_dims dimensions and Q and K have output_dims dimensions
    """
    #-------------------------------------------------------------------
    query_fn: eqx.Module
    key_fn: eqx.Module
    #-------------------------------------------------------------------

    def __init(self, input_dims, output_dims, use_bias=False, key=None):

        self.query_fn = nn.Linear(input_dims, output_dims, use_bias=use_bias, key=key)
        self.key_fn = nn.Linear(input_dims, output_dims, use_bias=use_bias, key=key)

    #-------------------------------------------------------------------

    def __call__(self, x):
        return self.query_fn(x), self.key_fn(x)

    #-------------------------------------------------------------------

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
        5. Sample a target j* for each generator node i with:
            j* ~ softmax(sij) if mode is "soft"
            or
            j* = argmax(sij) if mode is "hard"
        6. Add a new edge eij* for each generating node i with random embedding
            
    """
    #-------------------------------------------------------------------
    query_key_fn: t.Callable    
    prob_fn: t.Callable
    threshold_fn: t.Optional[t.Callable]
    threshold: t.Optional[float]
    score_fn: t.Callable
    select_mode: str
    self_loops: bool
    #-------------------------------------------------------------------

    def __init__(self, 
                 prob_fn: t.Callable, 
                 query_key_fn: t.Callable, 
                 score_fn: t.Callable = cosine_similarity, 
                 select_mode: str = "softmax",
                 threshold_fn: t.Optional[t.Callable] = None,
                 threshold: t.Optional[float] = 0.,
                 self_loops: bool = False):
        
        self.query_key_fn = query_key_fn
        self.prob_fn = prob_fn
        self.score_fn = score_fn
        self.self_loops = self_loops
        self.select_mode = select_mode
        self.threshold_fn = threshold_fn
        self.threshold = threshold
        
    #-------------------------------------------------------------------

    @ejit
    def __call__(self, graph: GGraph, key: jr.PRNGKey):
        
        key_prob, key_edges, key_samp = jr.split(key, 3)
        nodes, edges, rec, send, anodes, aedges, *_ = graph
        max_nodes, max_edges = nodes.shape[0], edges.shape[0]
        nids = jnp.arange(max_nodes)
        eids = jnp.arange(max_edges)
        n_active = anodes.sum().astype(int)
        e_active = aedges.sum().astype(int)
        
        # 1. Get dividing nodes
        probs = jax.vmap(self.prob_fn)(nodes)[..., 0]
        gens = jr.uniform(key_prob, (max_nodes,)) < (probs * anodes)
        gens = gens.astype(float)

        # 2. Compute scores
        Q, K = evmap(self.query_key_fn)(nodes)
        scores = evmap(evmap(self.score_fn, in_axes=(None, 0)), in_axes=(0, None))(Q, K)
        scores = jnp.clip(scores, -1e4, 1e4)
        scores = jnp.where(anodes[None, :], scores, -1e10)
        if not self.self_loops:
            scores = jnp.where(jnp.identity(max_nodes), -1e10, scores)
        
        # 3. Sample receivers
        if self.select_mode == "threshold":
            #raise NotImplementedError
            if self.threshold_fn is not None:
                threshold = jax.vmap(self.threshold_fn)(nodes)
            else :
                threshold = self.threshold
            gens = jnp.logical_and(scores>threshold, gens[:, None])
            select = jnp.stack([jnp.arange(max_nodes)]*max_nodes)
            def add_edge(c, x):
                g, k = c
                nk, sk = jr.split(k)
                ge, se = x
                ng = self.add_edges(g, ge, se, sk)
                return [ng, nk], None

            [new_graph, _], _ = jax.lax.scan(
                lambda c, x: jax.lax.cond(
                    (x[0]>0).any(),
                    add_edge,
                    lambda c, x: (c, None),
                    c, x
                ),
                [graph, key_edges],
                [gens.T, select.T]
            )
            return new_graph

        gens = gens * (scores.max(-1)>self.threshold).astype(float)
        if self.select_mode == "softmax":
            select = jnp.where(gens, jr.categorical(key_samp, scores, axis=-1).astype(int), 0) 
        elif self.select_mode == "hardmax":
            select = jnp.where(gens, jnp.argmax(scores, axis=-1).astype(int), 0)

        return jax.lax.cond(
            (gens>0).any(),
            self.add_edges,
            lambda graph, gens, select, key: graph,
            graph, gens, select, key_edges
        )

    #-------------------------------------------------------------------

    def add_edges(self, graph, gens, select, key):

        nodes, edges, rec, send, anodes, aedges, *_ = graph
        max_nodes, max_edges = nodes.shape[0], edges.shape[0]
        nids = jnp.arange(max_nodes)
        eids = jnp.arange(max_edges)
        n_active = anodes.sum().astype(int)
        e_active = aedges.sum().astype(int)
        mat_e = incr_matrix(max_edges)
        def incr_edges(aedges, n):
            return jax.lax.fori_loop(
                0, n, lambda i, x: jnp.clip(x @ mat_e, 0., 1.), aedges
            ).at[-1].set(0.)

        # 4. Check if edge exist
        is_s = nids[:, None]==send[None, :]# (n, e)
        is_r = select[:, None]==rec[None, :]
        exist = jnp.logical_and(is_s, is_r).any(-1) & gens.astype(bool)
        gens = jnp.where(exist, 0., gens)
        
        # 5. Add new edges
        allowed = max_edges - e_active - 1
        n_gens = jnp.clip(gens.astype(int).sum(), 0, allowed)
        naedges = incr_edges(aedges, n_gens)
        mask_new_edges = naedges * (1-aedges)
        new_edges = edges + jr.normal(key, edges.shape) * mask_new_edges[..., None]
        
        # 6. Add new senders
        trgets = (jnp.cumsum(gens) * gens - 1).astype(int) + (e_active * gens).astype(int)
        nsend = jax.ops.segment_sum(nids, trgets, max_edges)
        nsend = jnp.where(mask_new_edges, nsend, send)

        # 7. Add receivers
        trgets = jnp.cumsum(gens) * gens - gens
        trgets = jnp.where(gens, trgets.astype(int), -1) + e_active * gens.astype(int)
        nrec = jax.ops.segment_sum(select, trgets, max_edges)
        nrec = jnp.where(mask_new_edges, nrec, rec)

        nrec = jnp.where(naedges, nrec, max_nodes-1)
        nsend = jnp.where(naedges, nsend, max_nodes-1)

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
        nodes, edges, rec, send, anodes, aedges, *_ = graph

        node_rec = nodes[rec]
        node_send = nodes[send]
        probs = jax.vmap(self.prob_fn)(jnp.concatenate([node_rec, node_send]))[:,0] * aedges
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










