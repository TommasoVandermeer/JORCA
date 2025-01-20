import jax.numpy as jnp
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib import use

# TODO: Add generate random humans parameters function
# TODO: Add generate random circular crossing scenario initial conditions function
# TODO: Add generate random parallel traffica initial conditions function

def get_standard_humans_parameters(n_humans:int) -> jnp.ndarray:
    """
    Returns the standard parameters of the ORCA for the humans in the simulation. Parameters are the same for all humans in the form:
    (radius, time_horizon, v_max, safety_space).

    args:
    - n_humans: int - Number of humans in the simulation.

    outputs:
    - parameters (n_humans, 4) - Standard parameters for the humans in the simulation.
    """
    single_params = jnp.array([0.3, 5, 1., 0.01])
    return jnp.tile(single_params, (n_humans, 1))

def plot_state(
        ax:Axes, 
        time:float, 
        full_state:jnp.ndarray,  
        humans_radiuses:np.ndarray, 
        plot_time=True) -> None:
    """
    """
    colors = list(mcolors.TABLEAU_COLORS.values())
    num = int(time) if (time).is_integer() else (time)
    ax.set(xlabel='X',ylabel='Y',xlim=[-9,9],ylim=[-9,9])
    # Humans
    for h in range(len(full_state)): 
        circle = plt.Circle((full_state[h,0],full_state[h,1]),humans_radiuses[h], edgecolor=colors[h%len(colors)], facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
        if plot_time: ax.text(full_state[h,0],full_state[h,1], f"{num}", color=colors[h%len(colors)], va="center", ha="center", size=10 if (time).is_integer() else 6, zorder=1, weight='bold')
        else: ax.text(full_state[h,0],full_state[h,1], f"{h}", color=colors[h%len(colors)], va="center", ha="center", size=10, zorder=1, weight='bold')

def animate_trajectory(
    states:jnp.ndarray, 
    humans_radiuses:np.ndarray, 
    dt:float=0.25,
    save:bool=False,
    save_dir:str=None,
    ) -> None:

    # TODO: Add a progress bar,
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.78, top=0.90, bottom=0.05)
    ax.set_aspect('equal')
    ax.set(xlim=[-10,10],ylim=[-10,10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    def animate(frame):
        ax.clear()
        ax.set_title(f"Time: {'{:.2f}'.format(round(frame*dt,2))} - Humans policy: ORCA", weight='bold')
        plot_state(ax, frame*dt, states[frame], humans_radiuses, plot_time=False)

    anim = FuncAnimation(fig, animate, interval=dt*1000, frames=len(states))
    
    anim.paused = False
    def toggle_pause(self, *args, **kwargs):
        if anim.paused: anim.resume()
        else: anim.pause()
        anim.paused = not anim.paused

    fig.canvas.mpl_connect('button_press_event', toggle_pause)
    plt.show()

    if save:
        use('Agg')
        anim.save(save_dir, writer='imagemagick', fps=10)