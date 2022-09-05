import os
import copy
import numpy as np
from agents import WorkerAgent, Agent
import gym
import argparse # Would like to use later
from utils import plot_learning_curve
import matplotlib.pyplot as plt


if __name__ == "__main__":

    num_games = 50
    episode_length = 10**6
    option_interval = 3
    env = gym.make("-------------")
    batch_size = 100
    skill_dims = 4

    plot_dir = "./plots/"
    checkpoint_dir = "./checkpoints/"

    agent = Agent(env=env, 
                  skill_dims=skill_dims, 
                  learning_rate=10**-4, 
                  memory_size=10**6, 
                  checkpoint_dir=checkpoint_dir, 
                  option_interval=option_interval, 
                  gamma=0.99, 
                  option_gamma=0.99, 
                  reward_scale=2, 
                  batch_size=batch_size, 
                  polyak_coeff=0.999, 
                  beta=0.01, 
                  alpha=0.2, 
                  use_auto_entropy_adjustment=True, 
                  min_entropy_target=0.02, 
                  w_alpha=0.01, 
                  w_auto_entropy_adjustment=False, 
                  use_tanh=True, 
                  feature="action")

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    worker_score_history = []

    if not load_checkpoint:

        for i in range(num_games):
            observation = env.reset()
            worker_score = 0
            plt.figure()

            print("*************--Starting data collection--*************")
            for j in range(episode_length):
                skill = agent.generate_skill(state=observation)
                states_array, actions_array, next_states_array, rewards_array, actor_log_probs_array, done_array = agent.generate_empty_arrays()

                for k in range(option_interval):
                    action, actor_log_probs = agent.worker.choose_action(state=observation, skill=skill)
                    next_observation, reward, done, info = env.step(action)
                    states_array[k, :] = observation
                    actions_array[k, :] = action
                    next_states_array[k, :] = next_observation
                    rewards_array[k] = reward
                    actor_log_probs_array[k] = actor_log_probs.item()
                    done_array[k] = done

                agent.worker.remember(state_array=states_array, 
                                      action_array=actions_array, 
                                      next_state_array=next_states_array, 
                                      skill=skill, 
                                      reward_array=rewards_array, 
                                      done_array=done_array)
                scheduler_reward = agent.post_interval_reward(actor_log_probs=actor_log_probs_array, 
                                                              reward_array=rewards_array, 
                                                              expected_value=True)
                agent.remember(state=observation, 
                               skill=skill, 
                               next_state=next_observation, 
                               reward=scheduler_reward, 
                               done=done)
                observation = next_observation
                
            
            # Model training
            print("*************--Starting model training--*************")
            for j in range(episode_length//batch_size):
                worker_reward = agent.learn()
                worker_score += worker_reward.detach().cpu().numpy()
                worker_score_history.append(worker_score)

            plt.plot(range(1, (episode_length//batch_size) + 1), worker_score_history)
        
        plt.savefig(plot_dir + env.spec.id + " Score history.png")
