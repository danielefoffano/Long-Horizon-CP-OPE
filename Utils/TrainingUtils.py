from Agents.Agent import Agent
from Utils.Experience import Experience
import numpy as np
import torch


def train_behaviour_policy(env, agent: Agent, MAX_STEPS):

    episode_rewards = []
    episode_steps = []

    episode = 0
    state, _ = env.reset()
    steps = 0
    while steps < MAX_STEPS:
        rewards = 0

        action = agent.forward(state, steps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.backward(Experience(state, action, reward, next_state, done))

        state = next_state

        steps += 1
        rewards += reward


        episode_rewards.append((episode, rewards))
        episode_steps.append((episode, steps))
        if done:
          state, _ = env.reset()
          episode += 1


    return agent.q_function

def train_weights(nu_net, zeta_net, n_steps, sa_buffer, next_sa_buffer, buffer_start, lr_nu, lr_zeta, DISCOUNT_FACTOR):
   
  optimizer_nu = torch.optim.Adam(nu_net.parameters(), lr_nu)
  optimizer_zeta = torch.optim.Adam(zeta_net.parameters(), lr_zeta)

  losses = []

  for step in range(n_steps):
    
    idx_sa = np.random.choice(len(sa_buffer), size=2048, replace=False)
    idx_start_s = np.random.choice(len(buffer_start), size=2048, replace=False)

    batch_sa = torch.as_tensor(sa_buffer[idx_sa], dtype=torch.float32)
    batch_next_sa = torch.as_tensor(next_sa_buffer[idx_sa], dtype=torch.float32)
    batch_start_s = torch.as_tensor(buffer_start[idx_start_s], dtype=torch.float32)

    optimizer_nu.zero_grad()
    optimizer_zeta.zero_grad()

    out_nu_sa = nu_net.forward(batch_sa)
    out_nu_next_sa = nu_net.forward(batch_next_sa)
    out_nu_start_s = nu_net.forward(batch_start_s)
    out_zeta_sa = zeta_net.forward(batch_sa)

    loss_nu = ((out_nu_sa - DISCOUNT_FACTOR*out_nu_next_sa)*out_zeta_sa.detach() - (1/3)*(out_zeta_sa.detach().abs()**3) -(1-DISCOUNT_FACTOR)*out_nu_start_s).mean()
    loss_zeta = -((out_nu_sa.detach() - DISCOUNT_FACTOR*out_nu_next_sa.detach())*out_zeta_sa - (1/3)*(out_zeta_sa.abs()**3) -(1-DISCOUNT_FACTOR)*out_nu_start_s.detach()).mean()

    loss_nu.backward()
    loss_zeta.backward()
    optimizer_nu.step()
    optimizer_zeta.step()

    losses.append(loss_zeta.item())

    print(f"Iteration {step} - Zeta Loss: {loss_zeta.item()}")

  return nu_net, zeta_net, losses