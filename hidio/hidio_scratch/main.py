import os
import copy
import numpy as np
from agents import WorkerAgent, Agent
import gym
import argparse # Would like to use later
from utils import plot_learning_curve

# Another scheduler training strategy could be to train via log_probs

if __name__ == "__main__":
    env = gym.make("----------")
    option_interval = 3
    batch_size = 256

    plot_dir = "./plots/"

    agent = Agent(env=env, 
                  skill_dims=4, 
                  learning_rate=10**-4, 
                  memory_size=10**6, 
                  checkpoint_dir="./checkpoints/", 
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

    num_games = 150
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    worker_score_history = []

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    for j in range(num_games):
        observation = env.reset()
        done = False
        score = 0
        worker_score = 0


        while not done:
            skill = agent.generate_skill(state=observation)
            states_array, actions_array, next_states_array, rewards_array, actor_log_probs_array, done_array = agent.generate_empty_arrays()
            
            for i in range(option_interval):
                action, actor_log_probs = agent.worker.choose_action(state=observation, skill=skill)
                next_observation, reward, done, info = env.step(action)
                states_array[i, :] = observation
                actions_array[i, :] = action
                next_states_array[i, :] = next_observation
                rewards_array[i] = reward
                actor_log_probs_array[i] = actor_log_probs.item()
                done_array[i] = done

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
            
            score = score + scheduler_reward

            if not load_checkpoint:
                if agent.scheduler_memory.memory_counter > batch_size:
                    worker_reward = agent.learn()
                    worker_score = worker_score + worker_reward.detach().cpu().numpy()
                else:
                    agent.learn()

            observation = next_observation
            
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        if not load_checkpoint:
            worker_score_history.append(worker_score)
            avg_worker_score = np.mean(worker_score_history[-100:])
            print("Episode = {}, Score = {:.1f}, Average score = {:.2f}, Worker score = {:.3f}, Average worker score = {:.3f}".format(j, score, avg_score, worker_score, avg_worker_score))

        else:
            print("Episode = {}, Score = {:.1f}, Average score = {:.2f}".format(j, score, avg_score))

    if not load_checkpoint:
        plot_learning_curve(num_games=num_games, 
                            scheduler_scores=score_history, 
                            worker_scores=worker_score_history, 
                            plot_dir=plot_dir, 
                            env_name= env.spec.id, 
                            file_type=".png")