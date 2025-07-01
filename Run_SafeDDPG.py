import os
import numpy as np
import torch
import random
from squaremap import SquareMapEnv
from DDPG_base import DDPG, GCDDPG, DDPGLagrange
import datetime
import pickle
from torch.utils.tensorboard import SummaryWriter


def safeddpg(Alg_name, scale_mode, seed):

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)

    set_seed(seed)

    now = datetime.datetime.now()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    result_path = 'results'

    run_name = now.strftime("%Y%m%d_%H%M_%S") + "_scale" + \
        str(scale_mode) + '_' + Alg_name + "_seed" + str(seed)

    file_path = os.path.join(result_path, run_name)
    os.makedirs(file_path, exist_ok=True)

    process_path = os.path.join(file_path, "process.txt")
    history_path = os.path.join(file_path, "history.dat")
    model_path = os.path.join(file_path, 'model.pkl')
    writer = SummaryWriter(file_path)

    with open(process_path, mode='w') as file:
        if 'Lagrange' in Alg_name:
            file.write(
                f"{'Epoch':<26}{'EpochReward':<26}{'EpochCost':<26}{'EpReward':<26}{'epCost':<26}{'TimeStep':<26}{'Episode':<26}{'lagrange_multiplier':<26}\n")
        else:
            file.write(
                f"{'Epoch':<26}{'EpochReward':<26}{'EpochCost':<26}{'EpReward':<26}{'epCost':<26}{'TimeStep':<26}{'Episode':<26}\n")

    if Alg_name == 'DDPGRS':
        agent = DDPG(state_dim=4, action_dim=2, device=device)
    else:
        agent_class = globals()[Alg_name]
        agent = agent_class(state_dim=4, action_dim=2, device=device)

    env = SquareMapEnv(scale_mode)

    epochs = 100
    local_epoch_step = 5000
    max_ep_len = 100
    learning_starts = 10000

    if 'Lagrange' in Alg_name:
        agent.lagrange_multiplier = 0.1
        lagrange_multiplier_lr = 3e-4
        lagrange_multiplier_max = 100

    cost_lim = 0
    global_episode = 0
    global_step = 0
    epsilon_decay = 0.2
    decay_rate = epsilon_decay/epochs
    Q_lr_decay_rate = agent.Q_lr/epochs
    pi_lr_decay_rate = agent.pi_lr/epochs

    for epoch in range(epochs):

        if global_step <= learning_starts-1:
            epsilon = 1
        else:
            epsilon = epsilon_decay
            epsilon_decay -= decay_rate

            agent.critic_optimizer.param_groups[0]['lr'] -= Q_lr_decay_rate
            agent.pi_lr -= pi_lr_decay_rate
            agent.actor_optimizer.param_groups[0]['lr'] = agent.pi_lr
            if 'GCDDPG' in Alg_name or 'Lagrange' in Alg_name:
                agent.cost_critic_optimizer.param_groups[0]['lr'] -= Q_lr_decay_rate
            if 'GCDDPG' in Alg_name:
                agent.cost_actor_optimizer.param_groups[0]['lr'] = agent.pi_lr

        state, ep_len, episode_cost, episode_reward, episode_path = env.reset(), 0, 0, 0, []
        state = torch.tensor(state).float().to(device)

        epoch_reward = []
        epoch_cost = []

        for t in range(local_epoch_step):

            with torch.no_grad():
                if 'GCDDPG' in Alg_name:
                    action1, action2 = agent.act(
                        state, epsilon=epsilon)
                    action1 += torch.randn_like(action1) * epsilon_decay
                    action2 += torch.randn_like(action2) * epsilon_decay
                    action = action1 + action2
                else:
                    action = agent.act(state, epsilon=epsilon)
                    action += torch.randn_like(action) * epsilon_decay

            next_state, reward, cost, done, _ = env.step(action.cpu().numpy())

            episode_path.append(next_state)
            episode_reward += reward
            episode_cost += cost

            next_state = torch.tensor(next_state).float().to(device)
            reward = torch.tensor(reward).float().to(device)
            cost = torch.tensor(cost).float().to(device)

            if 'GCDDPG' in Alg_name:
                agent.remember(state, action1, action2,
                               reward, cost, next_state, done)
            elif 'Lagrange' in Alg_name:
                agent.remember(state, action, reward, cost, next_state, done)
            elif 'RS' in Alg_name:
                agent.remember(state, action, (reward-cost), next_state,
                               done)
            else:
                agent.remember(state, action, reward, next_state, done)

            if global_step > learning_starts:
                transitions = random.sample(agent.memory, agent.batch_size)
                critic_loss, cost_critic_loss = agent.learn_critic(transitions)

                if global_step % agent.policy_frequency == 0:
                    if 'GCDDPG' in Alg_name:
                        if last_episode_cost < cost_lim:
                            agent.actor_optimizer.param_groups[0]['lr'] = 1.1 * \
                                agent.pi_lr
                            agent.cost_actor_optimizer.param_groups[0]['lr'] = agent.pi_lr
                        else:
                            agent.actor_optimizer.param_groups[0]['lr'] = agent.pi_lr
                            agent.cost_actor_optimizer.param_groups[0]['lr'] = 1.1 * agent.pi_lr
                    actor_loss, cost_actor_loss = agent.learn_actor(
                        transitions)

                if global_step % agent.target_frequency == 0:
                    agent.soft_update(agent.target_critic, agent.critic)
                    agent.soft_update(agent.target_actor, agent.actor)
                    if global_step % 100 == 0:
                        writer.add_scalar("losses/critic_loss",
                                          critic_loss.item(), global_step)
                        writer.add_scalar("losses/actor_loss",
                                          actor_loss.item(), global_step)
                    if 'GCDDPG' in Alg_name or 'Lagrange' in Alg_name:
                        agent.soft_update(
                            agent.target_cost_critic, agent.cost_critic)
                        if global_step % 100 == 0:
                            writer.add_scalar("losses/cost_critic_loss",
                                              cost_critic_loss.item(), global_step)
                    if 'GCDDPG' in Alg_name:
                        agent.soft_update(
                            agent.target_cost_actor, agent.cost_actor)
                        if global_step % 100 == 0:
                            writer.add_scalar("losses/cost_actor_loss",
                                              cost_actor_loss.item(), global_step)

            state = next_state

            global_step += 1
            ep_len += 1

            if done or ep_len == max_ep_len-1 or t == local_epoch_step-1:
                with open(history_path, 'ab') as file:
                    pickle.dump(episode_path, file)
                print(
                    f"Epoch {epoch}, Episode {global_episode}, Reward {episode_reward}, Cost {episode_cost}")
                writer.add_scalar("charts/episode_reward",
                                  episode_reward, global_step)
                writer.add_scalar("charts/episode_cost",
                                  episode_cost, global_step)

                global_episode += 1
                last_episode_cost = episode_cost
                epoch_reward.append(episode_reward)
                epoch_cost.append(episode_cost)

                state, ep_len, episode_cost, episode_reward, episode_path = env.reset(), 0, 0, 0, []
                state = torch.tensor(state).float().to(device)

        mean_ep_cost = np.mean(epoch_cost)
        mean_ep_reward = np.mean(epoch_reward)
        if 'Lagrange' in Alg_name:
            if global_step > learning_starts:
                agent.lagrange_multiplier += lagrange_multiplier_lr * \
                    (mean_ep_cost - cost_lim)
                if agent.lagrange_multiplier < 0:
                    agent.lagrange_multiplier = 0
                elif agent.lagrange_multiplier > lagrange_multiplier_max:
                    agent.lagrange_multiplier = lagrange_multiplier_max

            with open(process_path, mode='a') as file:
                file.write(
                    f"{epoch:<26}{np.sum(epoch_reward):<26}{np.sum(epoch_cost):<26}{mean_ep_reward:<26}{mean_ep_cost:<26}{global_step:<26}{global_episode:<26}{agent.lagrange_multiplier:<26}\n")
        else:
            with open(process_path, mode='a') as file:
                file.write(
                    f"{epoch:<26}{np.sum(epoch_reward):<26}{np.sum(epoch_cost):<26}{mean_ep_reward:<26}{mean_ep_cost:<26}{global_step:<26}{global_episode:<26}\n")

        if 'GCDDPG' in Alg_name:
            state = {
                'actor_state_dict': agent.actor.state_dict(),
                'cost_actor_state_dict': agent.cost_actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'cost_critic_state_dict': agent.cost_critic.state_dict()
            }
        elif 'Lagrange' in Alg_name:
            state = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'cost_critic_state_dict': agent.cost_critic.state_dict()
            }
        else:
            state = {
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict()
            }
        torch.save(state, model_path)


seed = 0

# DDPG, DDPGRS(reward shaping), DDPGLagrange, GCDDPG(GCPO)
Alg_name = 'GCDDPG'

# | Mode | Description                |
# |------|----------------------------|
# |  1   | r_target=1, c_cost=1       |
# |  2   | r_target=10, c_cost=1      |
# |  3   | r_target=100, c_cost=0.1   |
scale_mode = 1

safeddpg(Alg_name, scale_mode, seed)
