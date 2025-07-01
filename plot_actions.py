import matplotlib.pyplot as plt
import numpy as np
from squaremap import SquareMapEnv
import torch
from DDPG_base import DDPG, GCDDPG, DDPGLagrange


filefold_paths = ['results/20240501_2150_19_scale2_GCDDPG_seed0',
                  ]

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
                         facecolor='#DB7E7B', alpha=1)
    ax.add_patch(rect)

for zone in maze.target_zones:
    x1, y1, x2, y2 = zone
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                         facecolor='#8CB57E', alpha=1)

    ax.add_patch(rect)

plt.scatter(maze.state[0]*10, maze.state[1]*10,
            color='#0072BD', s=100, label='Start point', alpha=0.5)

episode_paths = []

for filefold_path in filefold_paths:

    with open(filefold_path + '/model.pkl', 'rb') as file:
        state = torch.load(file)

    if 'GC' in filefold_path:
        agent = GCDDPG(state_dim=4, action_dim=2)
    elif 'Lagrange' in filefold_path:
        agent = DDPGLagrange(state_dim=4, action_dim=2)
    elif 'RS' in filefold_path:
        agent = DDPG(state_dim=4, action_dim=2)

    agent.critic.load_state_dict(state['critic_state_dict'])
    critic = agent.critic
    agent.actor.load_state_dict(state['actor_state_dict'])
    if 'GC' in filefold_path or 'Lagrange' in filefold_path:
        agent.cost_critic.load_state_dict(state['cost_critic_state_dict'])
        cost_critic = agent.cost_critic
    if 'GC' in filefold_path:
        agent.cost_actor.load_state_dict(state['cost_actor_state_dict'])

    state, episode_path, done = maze.reset(), [], False
    state = torch.tensor(state).float()
    i = 0
    while done is not True and i <= 100:
        with torch.no_grad():
            action1, action2 = agent.act(state)
            action = action1 + action2
        next_state, reward, cost, done, _ = maze.step(action)

        action1 = maze.max_acceleration * action1
        action2 = maze.max_acceleration * action2
        action = maze.max_acceleration * action

        step_path = np.concatenate(
            (next_state, action, action1, action2), axis=0)

        episode_path.append(step_path)

        state = next_state
        state = torch.tensor(state).float()
        i += 1

    episode_paths.append(episode_path)

for episode_path in episode_paths:
    x = []
    y = []
    for step_path in episode_path:
        x.append(step_path[0]*10)
        y.append(step_path[1]*10)
    plt.plot(x, y, color='#0072BD', linewidth=3, alpha=0.2)

for episode_path in episode_paths:
    for step_path in episode_path:
        if 'GC' in filefold_path:
            plt.quiver(step_path[0]*10, step_path[1]*10, step_path[6], step_path[7],
                       angles='xy', scale_units='xy', scale=0.5, width=0.006, color='#76A765', alpha=0.6)
            plt.quiver(step_path[0]*10, step_path[1]*10, step_path[8], step_path[9],
                       angles='xy', scale_units='xy', scale=0.5, width=0.006, color='#D35E5B', alpha=0.6)

        plt.quiver(step_path[0]*10, step_path[1]*10, step_path[4], step_path[5],
                   angles='xy', scale_units='xy', scale=0.5, width=0.006, color='#0072BD', alpha=0.8)


plt.savefig('plot_result/' + 'actions' + '.png', dpi=300, bbox_inches='tight')
plt.show()
