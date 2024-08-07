import gymnasium as gym
import torch.nn as nn

from Agents.QLearningAgent import QlearningAgent
from Utils.TrainingUtils import train_behaviour_policy, train_weights
from Policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from Networks.MLP import MLP
from Utils.CollectDataset import CollectSADatasets, CollectStartDataset
from Utils.WeightsEvaluation import evaluate_weights
from Utils.PlotLoss import plot_loss


#env = gym.make("FrozenLake-v1")
env = gym.make('Taxi-v3')

ALPHA = 0.6
DISCOUNT_FACTOR = 0.995
NUM_STATES = env.observation_space.n
NUM_ACTIONS = env.action_space.n
NUM_STEPS_BEHAVIOUR_TRAINING = 1000000
N_TRAJECTORIES_DATASET = 10000
EVAL_TRAJECTORIES = 1000

start_samples = 500000
encoding_size = 2
input_size = encoding_size + 1 #because action is not encoded, otherwise encoding_size*2
lr_nu = 0.0001
lr_zeta = 0.0001
weights_training_steps = 40000

agent = QlearningAgent(NUM_STATES, NUM_ACTIONS, DISCOUNT_FACTOR, ALPHA)

print("Computing Q-Learning solution")
q_function = train_behaviour_policy(env, agent, NUM_STEPS_BEHAVIOUR_TRAINING)

behaviour_p = EpsilonGreedyPolicy(q_function, 0.7, NUM_ACTIONS)
target_p = EpsilonGreedyPolicy(q_function, 0.1, NUM_ACTIONS)

state_encoding = nn.Embedding(500, encoding_size)
action_encoding = None
#action_encoding = nn.Embedding(6, encoding_size)

print("Collecting SA and S'A' datasets")
sa_buffer, next_sa_buffer = CollectSADatasets(env, behaviour_p, target_p, state_encoding, action_encoding, N_TRAJECTORIES_DATASET)

print("Collecting Starting States dataset")
buffer_start = CollectStartDataset(env, target_p, state_encoding, action_encoding, start_samples)

nu_net = MLP(input_size, 64, 1, False, False)
zeta_net = MLP(input_size, 64, 1, False, True)

print("Training Weights Network")
nu_net, zeta_net, losses = train_weights(nu_net, zeta_net, weights_training_steps, sa_buffer, next_sa_buffer, buffer_start, lr_nu, lr_zeta, DISCOUNT_FACTOR)

evaluate_weights(env, behaviour_p, target_p, state_encoding, action_encoding, zeta_net, EVAL_TRAJECTORIES)

plot_loss(losses, weights_training_steps)