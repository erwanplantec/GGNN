import jax.numpy as jnp
import jax
import jax.random as jr

import jraph
import equinox as eqx
import equinox.nn as nn
from equinox import filter_jit as ejit, filter_vmap as evmap, filter_pmap as epmap
from functools import partial
import networkx as nx

from src.utils import jraph_to_nx
from src.models import GGraph

@jax.jit
def in_degrees(g: GGraph):
    in_degrees = jax.ops.segment_sum(jnp.ones(g.receivers.shape), g.receivers, g.nodes.shape[0])
    return in_degrees * g.active_nodes

@jax.jit
def out_degrees(g: GGraph):
    out_degrees = jax.ops.segment_sum(jnp.ones(g.senders.shape), g.senders, g.nodes.shape[0])
    return out_degrees * g.active_nodes

@jax.jit
def degrees(g: GGraph):
    return in_degrees(g) + out_degrees(g)

@partial(jax.jit, static_argnames=("min_degree", "max_degree"))
def degree_distribution(g: GGraph, min_degree: int=0, max_degree: int=10):
    degs = jnp.clip(degrees(g).astype(int), min_degree, max_degree)
    degs = jnp.where(g.active_nodes, degs, -1)
    distr = jax.ops.segment_sum(jnp.ones((g.nodes.shape[0])),
                                degs, max_degree-min_degree)
    return distr / (distr.sum()+1e-8)

@partial(jax.jit, static_argnames=("min_degree", "max_degree"))
def out_degree_distribution(g: GGraph, min_degree: int=0, max_degree: int=10):
    degs = jnp.clip(out_degrees(g).astype(int), min_degree, max_degree)
    distr = jax.ops.segment_sum(jnp.ones((g.nodes.shape[0])),
                                degs, max_degree-min_degree)
    return distr / (distr.sum()+1e-8)

@partial(jax.jit, static_argnames=("min_degree", "max_degree"))
def in_degree_distribution(g: GGraph, min_degree: int=0, max_degree: int=10):
    degs = jnp.clip(in_degrees(g).astype(int), min_degree, max_degree)
    distr = jax.ops.segment_sum(jnp.ones((g.nodes.shape[0])),
                                degs, max_degree-min_degree)
    return distr / (distr.sum()+1e-8)

@jax.jit
def density(g: GGraph):
    m = g.active_edges.sum()
    n = g.active_nodes.sum()

    return m / (n*(n-1))

@jax.jit
def s_metric(g: GGraph):
    d = degrees(g)
    return jnp.sum(d[g.senders]*d[g.receivers])

@jax.jit
def avg_degree(g: GGraph):
    return jnp.sum(g.active_edges) / jnp.sum(g.active_nodes) 

def rich_club(nxg):
    _nxg = nxg.copy()
    _nxg.remove_edges_from(nx.selfloop_edges(_nxg))
    return nx.rich_club_coefficient(_nxg)

def n_louvain_communities(dnxg, **kwargs):
    return len(nx.louvain_communities(dnxg, **kwargs))


graph_metrics_fn = {
    "n_nodes":        lambda g, dnxg, nxg: jnp.sum(g.active_nodes),
    "n_edges":        lambda g, dnxg, nxg: jnp.sum(g.active_edges),
    "avg_degree":     lambda g, dnxg, nxg: avg_degree(g),
    "modularity":     lambda g, dnxg, nxg: nx.community.modularity(nxg, nx.community.greedy_modularity_communities(nxg)),
    "g_efficiency":   lambda g, dnxg, nxg: nx.global_efficiency(nxg),
    "l_efficiency":   lambda g, dnxg, nxg: nx.local_efficiency(nxg),
    "clustering":     lambda g, dnxg, nxg: nx.average_clustering(nxg),
    "transitivity":   lambda g, dnxg, nxg: nx.transitivity(nxg),
    "s_metric":       lambda g, dnxg, nxg: s_metric(g), #
    # "diameter":       lambda g, dnxg, nxg: nx.diameter(nxg), #Throw errors if not connected
    "flow_hierarchy": lambda g, dnxg, nxg: nx.flow_hierarchy(dnxg), #http://web.mit.edu/~cmagee/www/documents/28-DetectingEvolvingPatterns_FlowHierarchy.pdf
    #"non_randomness": lambda g, dnxg, nxg: nx.non_randomness(nxg)[1], #Error if not connected
    #"sw_omega":       lambda g, dnxg, nxg: nx.omega(nxg), #Slow and error if not connected
    "sw_sigma":       lambda g, dnxg, nxg: nx.sigma(nxg) if nx.is_connected(nxg) else 0., #Error if not connected
    "rich_club":      lambda g, dnxg, nxg: rich_club(nxg), #Error if self-loops
    # "wiener_index":   lambda g, dnxg, nxg: nx.wiener_index(nxg), #Can throw inf
    #"fractal_dims":   lambda g, dnxg, nxg: cbb(network(nxg), 1, True)
    "communities":    lambda g, dnxg, nxg: len(nx.connected_components(nxg)),
    "isolates":       lambda g, dnxg, nxg: nx.number_of_isolates(nxg),
    "assortativity":    lambda g, dnxg, nxg: nx.degree_pearson_correlation_coefficient(nxg)
}

graph_metrics_names = list(graph_metrics_fn.keys())
jax_graph_metrics = ["n_nodes", "n_edges", "avg_degree", "s_metric"]

def GraphMetrics(metrics: list = graph_metrics_names):
    
    n = len(metrics)

    if set(metrics).issubset(set(jax_graph_metrics)):
        n = len(metrics)
        def _apply(graph: GGraph):
            res = jnp.zeros((n,))
            for i, metric in enumerate(metrics):
                tmp = graph_metrics_fn[metric](graph, 0., 0.)
                res = res.at[i].set(tmp)
            return res
        return _apply

    def _apply(graph: GGraph):
        dnxg = jraph_to_nx(graph)
        nxg = dnxg.to_undirected()
        res = jnp.zeros((n,))
        for i, metric in enumerate(metrics):
            tmp = graph_metrics_fn[metric](graph, dnxg, nxg)
            res = res.at[i].set(tmp)
        return res
        
    return _apply



if __name__ == "__main__":
    from src.utils import RandomGraph
    from timeit import timeit

    graph = RandomGraph(jr.PRNGKey(15), 150, 2, 350, 2, 200, 400)
    metrics = ["modularity"]
    apply_fn = GraphMetrics(metrics)
    emb = apply_fn(graph)
    for i, name in enumerate(metrics):
        print(name, ': ', emb[i])




