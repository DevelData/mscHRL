import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve

if __name__ == "__main__":
    env = gym.make("InvertedPendulumBulletEnv-v0")
    #print("Action space shape:", env.action_space.shape)
    agent = Agent(input_dims=env.observation_space.shape, 
                  env=env, 
                  num_actions=env.action_space.shape[0])
    n_games = 250
    filename = "inverted_pendulum.png"
    figure_file = "./plots/" + filename
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    
    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        
        while not done:
            #print("Observation 1: ------------", observation)
            #print("Observation", observation.shape)
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            #print("Observation 2: ------------", observation)
            #print("Action: ------------", action)
            #print("Next observation: ------------", next_observation)
            #print("Reward: ------------", reward)
            #print("Done: ------------", done)
            agent.remember(observation, action, next_observation, reward, done)
            
            if not load_checkpoint:
                agent.learn()
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        
        print("Episode = {}, Score = {:.1f}, Average Score = {:.2f}".format(i, score, avg_score))
        
    if not load_checkpoint:
        x = [i for i in range(1, n_games)]
        plot_learning_curve(x, score_history, figure_file)
        