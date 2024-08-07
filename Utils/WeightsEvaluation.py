import numpy as np
import torch

def evaluate_weights(env, behaviour_p, target_p, state_encoding, action_encoding, zeta_net, N_TRAJECTORIES):
    rewards = []
    for n in range(N_TRAJECTORIES):
        done = False
        state, _ = env.reset()
        c_rew = 0
        t = 0
        while not done:
            action = target_p.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            c_rew += reward #(DISCOUNT_FACTOR**t)*reward
            state = next_state
            t+=1

            rewards.append(reward)

    avg_target_reward = np.mean(rewards)

    w_rewards = []
    for n in range(N_TRAJECTORIES):
        done = False
        state, _ = env.reset()

        while not done:
            action = behaviour_p.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            enc_s = state_encoding(torch.as_tensor(state, dtype = torch.int32)).detach()
            enc_a = torch.as_tensor([action], dtype = torch.float32)
            sa_pair = torch.cat((enc_s,enc_a), dim=0)
            w = zeta_net(sa_pair)

            state = next_state

            w_rewards.append(reward*w.item())

    avg_est_rew = np.mean(w_rewards)

    print(f"Average target reward: {avg_target_reward} | Average target reward est: {avg_est_rew}")