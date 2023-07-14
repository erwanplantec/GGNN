import jax
import jax.numpy as jnp
import jax.random as jr
import typing as t
import qdax as qd

default_emitter = None

class QdaxTrainer:

	"""
	Args:

	eval_fn (Callable): Genotype x RNGKey -> Fitness x Descr x ExtraData x RNGKey 
	"""
	#-------------------------------------------------------------------
	
	def __init__(self, eval_fn: t.Callable, emitter: qd.Emitter=default_emitter,
				 metrics: t.Iterable=qd.default_qd_metrics):
		
		pass

	#-------------------------------------------------------------------

	def __call__(self, key: jr.PRNGKey):
		
		pass