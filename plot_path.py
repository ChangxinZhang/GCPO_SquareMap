import numpy as np
from squaremap import SquareMapEnv
import pickle
import matplotlib.pyplot as plt


filefold_path = 'results/20240501_2150_19_scale2_GCDDPG_seed0'

with open(filefold_path + '/history.dat', mode='rb') as file:
    all_episodes = []
    while True:
        try:
            episode_path = pickle.load(file)
            all_episodes.append(episode_path)
        except EOFError:
            break


maze = SquareMapEnv(0)
maze.reset()

fig, ax = plt.subplots()
ax.set_xlim(0, maze.width)
ax.set_ylim(0, maze.height)
ax.set_aspect('equal')
ax.grid(False)

ax.set_facecolor('#F0F0F0')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.spines['top'].set_color('dimgrey')
ax.spines['top'].set_linewidth(5)
ax.spines['bottom'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_color('dimgrey')
ax.spines['left'].set_linewidth(5)
ax.spines['right'].set_color('dimgrey')
ax.spines['right'].set_linewidth(5)


for zone in maze.danger_zones:
    x1, y1, x2, y2 = zone
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                         facecolor='#DB7E7B', alpha=1, label='Hazardous region')
    ax.add_patch(rect)


for zone in maze.target_zones:
    x1, y1, x2, y2 = zone
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                         facecolor='#8CB57E', alpha=1, label='Target area')
    ax.add_patch(rect)

plt.scatter(maze.state[0]*10, maze.state[1]*10,
            color='#0072BD', s=100, label='Start point')


# uniformly sample 500 episodes
num_episodes_to_select = 500
total_episodes = len(all_episodes)
step = total_episodes / num_episodes_to_select
selected_indices = np.round(np.arange(0, total_episodes, step)).astype(int)
selected_episodes = [all_episodes[i] for i in selected_indices]


# selected_episodes = all_episodes[-500:] # last 500 episodes


for i, episode in enumerate(selected_episodes):
    x = []
    y = []
    for state in episode:
        if isinstance(state, (list, tuple)) and len(state) >= 2:
            x.append(state[0]*10)
            y.append(state[1]*10)
    ax.plot(x, y, color='#0072BD', linewidth=0.1)


plt.savefig('plot_result/' + 'paths' + '.png', dpi=300, bbox_inches='tight')
plt.show()
