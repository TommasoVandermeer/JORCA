from jax import numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os

from jorca.orca import step
from jorca.utils import *

# Hyperparameters
n_humans = 5
circle_radius = 10
dt = 0.01
end_time = 15
sample_plot_time = 3
frame_dt = 0.1
save_gif = False

# Initial conditions
humans_state = np.zeros((n_humans, 4))
humans_goal = np.zeros((n_humans, 2))
angle_width = (2 * jnp.pi) / (n_humans)
for i in range(n_humans):
    # State: (px, py, vx, vy)
    humans_state[i,0] = circle_radius * jnp.cos(i * angle_width) + jax.random.randint(jax.random.PRNGKey(i), (1,), 0, 10)[0] * 0.1
    humans_state[i,1] = circle_radius * jnp.sin(i * angle_width) + jax.random.randint(jax.random.PRNGKey(i+n_humans), (1,), 0, 10)[0] * 0.1
    humans_state[i,2] = 0
    humans_state[i,3] = 0
    # Goal: (gx, gy)
    humans_goal[i,0] = -humans_state[i,0]
    humans_goal[i,1] = -humans_state[i,1]
humans_state = jnp.array(humans_state)
humans_parameters = get_standard_humans_parameters(n_humans)
humans_goal = jnp.array(humans_goal)
# Obstacles
static_obstacles = jnp.array([[[[1000.,1000.],[1000.,1000.]]]]) # dummy obstacles

# Dummy step - Warm-up (we first compile the JIT functions to avoid counting compilation time later)
_ = step(humans_state, humans_goal, humans_parameters, static_obstacles, dt)

# Simulation 
steps = int(end_time/dt)
print(f"\nAvailable devices: {jax.devices()}\n")
print(f"Starting simulation... - Simulation time: {steps*dt} seconds\n")
start_time = time.time()
all_states = np.empty((steps+1, n_humans, 4), np.float32)
all_states[0] = humans_state
for i in range(steps):
    humans_state = step(humans_state, humans_goal, humans_parameters, static_obstacles, dt)
    all_states[i+1] = humans_state
end_time = time.time()
print("Simulation done! Computation time: ", end_time - start_time)
all_states = jax.device_get(all_states) # Transfer data from GPU to CPU for plotting (only at the end)

# Plot
COLORS = list(mcolors.TABLEAU_COLORS.values())
print("\nPlotting...")
figure, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')
ax.set(xlabel='X',ylabel='Y',xlim=[-circle_radius-1,circle_radius+1],ylim=[-circle_radius-1,circle_radius+1])
for h in range(n_humans): 
    ax.plot(all_states[:,h,0], all_states[:,h,1], color=COLORS[h%len(COLORS)], linewidth=0.5, zorder=0)
    ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=COLORS[h%len(COLORS)], zorder=2)
    for k in range(0,steps+1,int(sample_plot_time/dt)):
        plot_state(ax, k*dt, all_states[k], humans_parameters[:,0], plot_time=True)
for o in static_obstacles: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
figure.savefig(os.path.join(os.path.dirname(__file__),".images",f"example1.png"), format='png')
plt.show()

## Animate trajectory 
# (WARNING: To save animation as GIF you need to have ImageMagick installed)
# sudo apt install libpng-dev libjpeg-dev libtiff-dev
# sudo apt install imagemagick
all_states = all_states[::int(frame_dt/dt)] # downsample
animate_trajectory(
    all_states, 
    humans_parameters[:,0], 
    dt=frame_dt, 
    save=save_gif,
    save_dir=os.path.join(os.path.dirname(__file__),".images","example1.gif"))