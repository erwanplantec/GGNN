from src.models import GGraph

import jax
import jax.numpy as jnp
import jax.random as jr
import jraph
import networkx as nx
import matplotlib.pyplot as plt
import typing as t

def jraph_to_nx(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _,_, _, anodes, aedges = jraph_graph
    n_nodes = int(anodes.sum())
    n_edges = int(aedges.sum())
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(n_nodes):
            nx_graph.add_node(n)
    else:
        for n in range(n_nodes):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(n_edges):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(n_edges):
            nx_graph.add_edge(
                int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph

def RandomGraph(key, n_nodes, node_features, n_edges, edge_features, max_nodes, max_edges):
    
    key, node_key, edge_key, send_key, rec_key = jr.split(key, 5)
    
    anodes = jnp.zeros((max_nodes,)).at[:n_nodes].set(1.)
    nodes = jr.normal(node_key, (max_nodes, node_features)).at[n_nodes:, :].set(0.)

    aedges = jnp.zeros((max_edges,)).at[:n_edges].set(1.)
    senders = jr.randint(send_key, (max_edges, ), minval=0, maxval=n_nodes).at[n_edges:].set(max_nodes-1)
    receivers = jr.randint(rec_key, (max_edges, ), minval=0, maxval=n_nodes).at[n_edges:].set(max_nodes-1)
    edges = jr.normal(edge_key, (max_edges, edge_features)).at[n_edges:, :].set(0.)

    graph = GGraph(nodes=nodes,
                   senders=senders,
                   receivers=receivers,
                   edges=edges,
                   n_node=jnp.array([n_nodes]),
                   n_edge=jnp.array([n_edges]),
                   globals=None,
                   active_nodes=anodes,
                   active_edges=aedges)
    return graph



def draw_graph(jraph_graph: jraph.GraphsTuple, seed=None) -> None:
    nx_graph = jraph_to_nx(jraph_graph)
    pos = nx.spring_layout(nx_graph, seed=seed)
    print(pos)
    nx.draw(
        nx_graph, pos=pos, with_labels=True, node_size=80
    )
    
def draw_graph_emb(jraph_graph, **draw_kws):
    nodes, edges, receivers, senders, _,_, _, anodes, aedges = jraph_graph
    nx_graph = jraph_to_nx(jraph_graph)
    n_nodes = int(anodes.sum())
    pos = {i: nodes[i, :2] for i in range(n_nodes)}
    node_color = np.array([nx_graph.out_degree(i) for i in range(n_nodes)])
    node_color = node_color / node_color.max()
    node_size = np.array([nx_graph.in_degree(i) for i in range(n_nodes)])
    node_size = node_size / node_size.max() * 200 + 50

    nx.draw(
        nx_graph, pos, nodelist=list(range(n_nodes)), node_color=node_color, node_size=node_size, **draw_kws
    )
    
def draw_graphs(graphs: t.Iterable[GGraph], n: int, nrows: int, ncols: int, 
                draw_fn: t.Callable = nx.draw_spring, rmv_isolates: bool=True, 
                title_fn: t.Callable = lambda i: f"step {i}", **draw_kws):
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16))
    for n in range(n):
        i, j = n//ncols, n%ncols
        g = jax.tree_map(lambda x: x[n], graphs)
        nxg = jraph_to_nx(g).to_undirected()
        if rmv_isolates:
            nxg.remove_nodes_from(list(nx.isolates(nxg)))
        draw_fn(
            nxg, node_size=20, ax=axs[i, j], **draw_kws
        )
        axs[i, j].set_title(title_fn(n))
    plt.show()