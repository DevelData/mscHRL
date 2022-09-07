# main/sac_main.py
import sys
sys.path.append("../")

import gym
import pybullet_envs
import numpy as np
from agents.sac_agent import Agent
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime



if __name__ == "__main__":
    env = gym.make("----------")
    checkpoint_dir = "../network_checkpoints/sac/"
    num_games = 500
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    transfer_network_params = False
    transfer_network_path = "./"
    env_name = env.spec.id
    performance_info_path = checkpoint_dir + "/" + env_name + "/model_info/"
    
    agent = Agent(env=env, 
                  max_memory_size=10**6, 
                  reward_scale=2, 
                  batch_size=512, 
                  polyak_coeff=0.995, 
                  checkpoint_dir=checkpoint_dir, 
                  gamma=0.99, 
                  alpha=0.25, 
                  use_auto_entropy_adjustment=True, 
                  min_target_entropy=0.005,
                  learning_rate=3*10**-4)

    # Create performance_info_path if it doesn't exist
    Path(performance_info_path).mkdir(parents=True, exist_ok=True)

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    if transfer_network_params:
        agent.transfer_network_params(load_path=transfer_network_path)

    for i in range(num_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action, _ = agent.choose_action(state=observation, 
                                            reparameterize=False)
            next_observation, reward, done, info = env.step(action)
            score = score + reward
            agent.remember(state_array=observation, 
                           action_array=action, 
                           next_state_array=next_observation, 
                           reward_array=reward, 
                           done_array=done)
            
            if not load_checkpoint:
                agent.learn()

            observation = next_observation

        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        if avg_score > best_score:
            best_score = avg_score
            
            if not load_checkpoint:
                agent.save_models()

        print("Episode = {}, Score = {:.1f}, Average Score = {:.2f}".format(i, score, avg_score))
    
    
    if not load_checkpoint:
        time_info = datetime.now().strftime("%Y_%m_%d_%H%M")

        # Score history plot
        plt.title("Score history for SAC in {}".format(env_name))
        plt.plot(range(1, len(score_history) + 1), np.array(score_history), label="Score")
        plt.legend(loc="upper left")
        plt.savefig(performance_info_path + "score_history_{}.png".format(time_info))

        # Score history JSON file
        model_performance_info = {"score_history": score_history, 
                                  "num_sampled": agent.memory.memory_counter}
        with open(performance_info_path + "score_history_{}.json".format(time_info), mode="w") as score_history_json_file:
            json.dump(model_performance_info, score_history_json_file)