from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax.experimental import host_callback
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functools import partial

def progress_bar_scan(num_samples, message=None):
    "Progress bar for a JAX scan"
    if message is None:
        #message = f"Running for {num_samples:,} iterations"
        message=""
    tqdm_bars = {}

    print_rate = 5 # if you run the sampler for less than 20 iterations
    remainder = num_samples % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(num_samples))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != num_samples-remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == num_samples-remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == num_samples-1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x   
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan




def plot_scan(func, freq=100, getter=lambda x, y: (x, x, y), clear=True):
    xs = []
    ys = []
    def _plot():
        if clear:
            clear_output()
        plt.scatter(xs, ys)
        plt.show()
    def _save_and_plot(ixy):
        i, x, y = ixy
        xs.append(x)
        ys.append(y)
        if not i % freq and i:
            _plot() 
    def _func(carry, x):
        carry, y = func(carry, x)
        host_callback.call(
            _save_and_plot,
            getter(x, y)
        )
        return carry, y
    return _func

if __name__ == "__main__":

    @plot_scan
    def f(c, x):
        return c+1, c+1

    _ = jax.lax.scan(f, 0, jnp.arange(1000))























