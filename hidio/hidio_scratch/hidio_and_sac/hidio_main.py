import gym
import pybullet_envs
import numpy as np
from hidio_agent import Agent
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime



if __name__ == "__main__":
    env = gym.make("InvertedPendulumBulletEnv-v0")
    checkpoint_dir = "../network_checkpoints/hidio/"
    option_interval = 3
    batch_size = 512
    num_games = 25
    best_score = env.reward_range[0]
    score_history = []
    worker_score_history = []
    load_checkpoint = False
    transfer_network_params = False
    transfer_network_path = "./"
    env_name = env.spec.id
    performance_info_path = checkpoint_dir + "/" + env_name + "/model_info/"


    agent = Agent(env=env, 
                  skill_dims=3, 
                  learning_rate=10**-4, 
                  memory_size=10**6, 
                  checkpoint_dir=checkpoint_dir, 
                  option_interval=option_interval, 
                  gamma=0.99, 
                  option_gamma=0.99, 
                  reward_scale=2, 
                  batch_size=batch_size, 
                  polyak_coeff=0.995, 
                  beta=0.01, 
                  alpha=0.3, 
                  use_auto_entropy_adjustment=True,
                  w_alpha=0.25, 
                  w_auto_entropy_adjustment=True, 
                  use_tanh=True, 
                  feature="stateAction")

    # Create performance_info_path if it doesn't exist
    Path(performance_info_path).mkdir(parents=True, exist_ok=True)
    
    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    if transfer_network_params:
        agent.transfer_network_params(load_path=transfer_network_path)

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
                                                          expected_log_probs=False)
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
        avg_score = np.mean(score_history[-20:])

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
        time_info = datetime.now().strftime("%Y_%m_%d_%H%M")

        # Score history plot
        plt.title("Score history for HIDIO in {}".format(env.spec.id))
        plt.plot(range(1, len(score_history) + 1), np.array(score_history), label="Model Score")
        plt.plot(range(1, len(score_history) + 1), np.array(worker_score_history), label="Worker Score")
        plt.legend(loc="upper left")
        plt.savefig(performance_info_path + "score_history_{}.png".format(time_info))

        # Score history JSON file
        model_performance_info = {"score_history": score_history, 
                                  "worker_score_history": worker_score_history, 
                                  "num_sampled": agent.scheduler_memory.memory_counter}
        with open(performance_info_path + "score_history_{}.json".format(time_info), mode="w") as score_history_json_file:
            json.dump(model_performance_info, score_history_json_file)