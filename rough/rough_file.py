import pybullet_envs
import gym


if __name__ == "__main__":
    env = gym.make("HumanoidBulletEnv-v0")
    #"MountainCarContinuous-v0")
    print("Observation space shape:", env.observation_space.shape)
    print("Action space shape:", env.action_space.shape)
    print("Action space shape high:", env.action_space.high)
    print("Action space shape low:", env.action_space.low)
    print("Name:", env.spec.id)
    print("Name:", type(env.spec.id))
    obs, _ = env.reset()
    print("Observation:", obs)
    
    #env.step(env.action_space.sample())