import matplotlib.pyplot as plt
import numpy as np
from squaremap import SquareMapEnv
import torch
from DDPG_base import DDPG, GCDDPG, DDPGLagrange


filefold_path = 'results/20240501_2150_19_scale2_GCDDPG_seed0'

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


maze = SquareMapEnv(0)
maze.reset()

fig, ax = plt.subplots()
ax.set_xlim(0, maze.width)
ax.set_ylim(0, maze.height)
ax.set_aspect('equal')
ax.grid(False)

ax.set_facecolor('#F0F0F0')
ax.spines['top'].set_color('dimgrey')
ax.spines['top'].set_linewidth(5)
ax.spines['bottom'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_color('dimgrey')
ax.spines['left'].set_linewidth(5)
ax.spines['right'].set_color('dimgrey')
ax.spines['right'].set_linewidth(5)

area_size = 0.5

Q_values = np.zeros((int(maze.height / area_size),
                    int(maze.width / area_size)))
if 'GC' in filefold_path or 'Lagrange' in filefold_path:
    Qc_values = np.zeros((int(maze.height / area_size),
                          int(maze.width / area_size)))

i = 0.0
idx_i = 0
while i < maze.width:
    j = 0.0
    idx_j = 0
    while j < maze.height:
        center_pos = np.array(
            [(i + area_size/2)/10, (j + area_size/2)/10, 0, 0])
        state_tensor = torch.unsqueeze(torch.tensor(
            center_pos, dtype=torch.float32), dim=0)

        with torch.no_grad():
            if 'GC' in filefold_path:
                Q = critic(state_tensor, torch.tensor(
                    [[0, 0]]), torch.tensor([[0, 0]])).item()
                Qc = cost_critic(state_tensor, torch.tensor(
                    [[0, 0]]), torch.tensor([[0, 0]])).item()
            else:
                action = agent.act(state_tensor)
                Q = critic(state_tensor,  torch.tensor([[0, 0]])).item()
                if 'Lagrange' in filefold_path:
                    Qc = cost_critic(
                        state_tensor,  torch.tensor([[0, 0]])).item()

        start_pos = center_pos * 10

        Q_values[idx_j, idx_i] = Q
        if 'GC' in filefold_path or 'Lagrange' in filefold_path:
            Qc_values[idx_j, idx_i] = Qc

        j += area_size
        idx_j += 1
    i += area_size
    idx_i += 1

extent = [0, maze.width, 0, maze.height]
plt.imshow(Q_values, extent=extent, origin='lower',
           cmap='coolwarm', alpha=0.2, interpolation='bicubic')
plt.colorbar(label='Q value')
plt.savefig('plot_result/'+'hotmap_Q' + '.png', dpi=300, bbox_inches='tight')
plt.close()


if 'GC' in filefold_path or 'Lagrange' in filefold_path:
    plt.imshow(Qc_values, extent=extent, origin='lower',
               cmap='coolwarm', alpha=0.2, interpolation='bicubic')
    plt.colorbar(label='Qc value')
    plt.savefig('plot_result/'+'hotmap_Qc'+'.png',
                dpi=300, bbox_inches='tight')
    plt.close()
