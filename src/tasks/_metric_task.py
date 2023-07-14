from src.models import GGraph
from src.metrics import *
from src.utils import smap

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import host_callback as hcb

import typing as t


def MetricTask(statics: t.Collection, init_graph: GGraph, metrics: list,
               targets: jnp.array, grow_iters: int = 20, apply_all_timesteps: bool = True,
               weights: jnp.array=None, init: str="model", graph_generator: t.Callable=None):
    
    metric_fn = GraphMetrics(metrics)
    
    if apply_all_timesteps and weights is None:
        weights = jnp.ones((grow_iters,))

    #-----------------vmappable------------
    
    if set(metrics).issubset(set(jax_graph_metrics)):
        @partial(jax.vmap, in_axes=(None, 0))
        def _eval(key, params):
            model = eqx.combine(params, statics)
            key_m, key_g = jr.split(key)
            if init == "random":
                _graph = graph_generator(key_g)
            else:
                _graph = init_graph
            graph, graphs = model(_graph, key, grow_iters, return_traj=True, init=init=="model")
            
            if apply_all_timesteps:
                ms = jax.vmap(metric_fn)(graphs) #(steps, n_metrics)
                fit = jnp.sum((ms-targets[None, ...])**2, axis=-1) * weights #(steps,)
                fit = jnp.mean(fit)
            else:
                m = metric_fn(graph)
                fit = jnp.mean((m-targets)**2)
            return fit
        return _eval
    

    n=len(metrics)

    def callback(graph):
        m = metric_fn(graph)
        return m

    #-----------------Not vmappable------------

    def _metric_fn(graph):
        return hcb.call(callback, graph, result_shape=jnp.ones((n,)))
    if apply_all_timesteps:
        _metric_fn = smap(_metric_fn)

    #TODO Add apply all timesteps handling for host calls
    def sub_eval(key, params):
        model = eqx.combine(params, statics)
        if init=="model":
            graph, graphs = model(init_graph, key, grow_iters, return_traj=True, init=True)
        elif init=="random":
            key_m, key_g = jr.split(key)
            graph, graphs = model(graph_generator(key_g), 
                                  key_m, 
                                  grow_iters, 
                                  return_traj=True, 
                                  init=False)
        else: 
            graph, graphs = model(init_graph, key, grow_iters, return_traj=True, init=False)

        if apply_all_timesteps:
            ms = _metric_fn(graphs)
            fit = jnp.sum((ms-targets[None, ...])**2, axis=-1) * weights #(steps,)
            fit = jnp.mean(fit)
        else: 
            m = _metric_fn(graph)
            fit = jnp.mean((m-targets)**2)
        return key, fit
    
    def _eval(key, params):
        _, fits = jax.lax.scan(sub_eval, key, params)
        return fits
    
    return _eval