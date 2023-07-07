import jax
import jax.numpy as jnp
import jax.random as jr

def model_apply(model, partition_fn = eqx.is_array):
	"""create an apply function with signature 
	(params, input)->output from eqx model"""
	params, static = eqx.partition(model, partition_fn)
	def _apply(params, *args):
		mod = eqx.combine(params, static)
		return mod(*args)
	return _apply, params


def smap(f, n_args: int=1, mapped_args=None):

    if mapped_args is None:
        mapped_args = list(range(n_args))
    idxs = []
    ic, ix = 0, 0
    for i in range(n_args):
        if i in mapped_args:
            idxs.append(ix)
            ix+=1
        else:
            idxs.append(ic)
            ic+=1

    def f_scan(c, x):
        args = [x[idxs[i]] if i in mapped_args else c[idxs[i]] for i in range(n_args)]
        y = f(*args)
        return c, y

    def wrapped(*args):
        xs = [args[i] for i in range(n_args) if i in mapped_args]
        c = [args[i] for i in range(n_args) if i not in mapped_args]
        _, ys = jax.lax.scan(f_scan, c, xs)
        return ys

    return wrapped