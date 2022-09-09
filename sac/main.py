import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve

if __name__ == "__main__":
    env = gym.make("InvertedPendulumBulletEnv-v0")
    agent = Agent(input_dims=env.observation_space.shape,
                  env=env,
                  num_actions=env.action_space.shape[0])
    # 250 in original
    n_games = 250
    filename = "inverted_pendulum.png"
    figure_file = "./plots/" + filename

    #######################
    #env.reward_range = (0,1000)
    ##########################

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = True

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    for i in range(n_games):
        # Small workaround because gym is being funny
        observation = env.reset()
        #############################
        #try:
        #    observation = env.reset()
        #except (ValueError, AssertionError):
        #    observation = env.reset()
        ###############################

        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
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
        #x = [i for i in range(1, n_games+1)]
        plot_learning_curve(range(1, n_games+1), score_history, figure_file)
