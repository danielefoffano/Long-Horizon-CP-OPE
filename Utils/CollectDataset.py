import numpy as np
import torch
def CollectSADatasets (env, behaviour_p, target_p, state_encoding, action_encoding, N_TRAJECTORIES):
    sa_buffer = []
    next_sa_buffer = []
    state,_ = env.reset()
    done = False

    for n in range(N_TRAJECTORIES):

        done = False
        state, _ = env.reset()
        while not done:
            action = behaviour_p.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            enc_s = state_encoding(torch.as_tensor(state, dtype = torch.int32)).detach()
            #enc_a = action_encoding(torch.as_tensor(action, dtype = torch.int32)).detach()
            enc_a = torch.as_tensor([action], dtype = torch.float32)
            sa_pair = torch.cat((enc_s,enc_a), dim=0)

            enc_next_s = state_encoding(torch.as_tensor(next_state, dtype = torch.int32)).detach()
            #enc_next_a = action_encoding(torch.as_tensor(target_p.get_action(next_state), dtype = torch.int32)).detach()
            enc_next_a = torch.as_tensor([target_p.get_action(next_state)], dtype = torch.float32)
            next_sa_pair = torch.cat((enc_next_s,enc_next_a), dim=0)

            sa_buffer.append(np.array([sa_pair.numpy()]))
            next_sa_buffer.append(np.array([next_sa_pair.numpy()]))
            state = next_state

    sa_buffer = np.array(sa_buffer)
    next_sa_buffer = np.array(next_sa_buffer)
    return sa_buffer, next_sa_buffer

def CollectStartDataset(env, target_p, state_encoding, action_encoding, n_samples):

    buffer_start = []
    for n in range(500000):
        s, _ = env.reset()

        enc_start_s = state_encoding(torch.as_tensor(s, dtype = torch.int32)).detach()
        #enc_start_a = action_encoding(torch.as_tensor(target_p.get_action(s), dtype = torch.int32)).detach()
        enc_start_a = torch.as_tensor([target_p.get_action(s)], dtype = torch.float32)
        start_sa_pair = torch.cat((enc_start_s,enc_start_a), dim=0)

        buffer_start.append(start_sa_pair.numpy())

    buffer_start = np.array(buffer_start)
    return buffer_start