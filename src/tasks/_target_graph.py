from src.models import GGraph
from src.metrics import *
from src.utils import smap

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import host_callback as hcb

import typing as t

def hamming(x, y):
    return jnp.where(x==y, 0., 1.).sum()

def dist_matrix(x, y):
    return (jnp.square(x[:, None, :]-y).sum(-1)+1e-5)**0.5

def sorted_matrix(x):
    return dist_matrix(x, x).sort(1).sort(0)

def TargetGraphTask(statics: t.Collection, init_graph: GGraph, target_graph: GGraph, 
                    grow_iters: int=100, targets: str="n", use_sorted_matrix: bool = False,
                    model_init: bool=False):

    nfeats = target_graph.nodes.shape[-1]
    efeats = target_graph.edges.shape[-1]

    if use_sorted_matrix and "n" in targets: 
        M_target = sorted_matrix(target_graph.nodes)

    @partial(jax.vmap, in_axes=(None, 0))
    def _eval(key, params):
        
        model = eqx.combine(params, statics)
        graph, graphs = model(init_graph, key, grow_iters, return_traj=True, init=model_init)

        loss = 0.
        
        if "n" in targets:
            if not use_sorted_matrix:
                node_loss = jnp.square(graph.nodes[:, :nfeats] - target_graph.nodes)
                node_loss = jnp.sum(node_loss * target_graph.active_nodes[..., None]) / target_graph.active_nodes.sum()
            else :
                M = sorted_matrix(graph.nodes[:, :nfeats])
                node_loss = jnp.mean(jnp.square(M - M_target))
            loss = loss + node_loss + (target_graph.active_nodes.sum() - graph.active_nodes.sum())**2
        
        if "e" in targets:
            edge_loss = jnp.mean(jnp.square(graph.edges[:, :efeats] - target_graph.edges))
            loss = loss + edge_loss
        
        return loss

    return _eval

def jin(x, v):
    return jnp.where((x==v).all(-1), True, False).any()

def jaxcard_dist(A, B):
    AinB = jax.vmap(jin, in_axes=(0, None))(A, B).astype(float).sum()
    return AinB / (A.shape[0]+B.shape[0] - AinB)


def TargetConnectomeTask(statics: t.Collection, init_graph: GGraph, target_graph: GGraph, 
                         grow_iters: int=100, model_init: bool=False):

    trec, tsend = target_graph.receivers, target_graph.senders
    tedges = jnp.concatenate((trec[:, None], tsend[:, None]), axis=-1)
    @partial(jax.vmap, in_axes=(None, 0))
    def _eval(key, params):
        model = eqx.combine(params, statics)
        graph, _ = model(init_graph, key, grow_iters, return_traj=True, init=model_init)
        rec, send = graph.receivers, graph.senders
        edges = jnp.concatenate((rec[:, None], semd[:, None]), axis=-1)

        dist = jaxcard_dist(edges, tedges)
        return diff

    return _eval



def LayeredGraph(n_nodes, n_edges, n_layers, max_nodes, max_edges, key, x_min: float=0., x_max: float=1.):
    assert n_nodes % n_layers == 0
        
    nodes = jnp.zeros((max_nodes, 2))
    anodes = jnp.zeros((max_nodes,)).at[:n_nodes].set(1.)
    npl = n_nodes // n_layers
    d = (x_max - x_min) / (n_layers-1)
    for l in range(n_layers):
        key, skey = jr.split(key)
        pos = l * d + x_min
        subnodes = jr.uniform(skey, (npl, 2)).at[:, 0].set(pos)
        nodes = nodes.at[l*npl: (l+1)*npl].set(subnodes)

    edges = jnp.zeros((max_edges, 1))
    aedges = jnp.zeros((max_edges,)).at[:n_edges].set(1.)
    rec = send = jnp.zeros((max_edges,)).astype(int)

    return  GGraph(nodes=nodes, edges=edges, active_nodes=anodes, active_edges=aedges, 
                   senders=send, receivers=rec)




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    g = LayeredGraph(20, 2, 4, 30, 30, jr.PRNGKey(1))
    print(g.nodes.shape)
    x, y = g.nodes.T
    plt.scatter(x, y)
    plt.show()








