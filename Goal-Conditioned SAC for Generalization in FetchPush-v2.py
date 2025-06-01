import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define training and test goals
TRAIN_GOAL_RANGE = [0.4, 0.6]  # Training goals sampled from this range for x, y
TEST_GOALS = [np.array([0.65, 0.65, 0.0]), np.array([0.7, 0.7, 0.0])]  # New goals outside training range
NUM_TRAIN_GOALS = 100

# Generate training goals
train_goals = [np.array([np.random.uniform(*TRAIN_GOAL_RANGE), np.random.uniform(*TRAIN_GOAL_RANGE), 0.0]) for _ in range(NUM_TRAIN_GOALS)]

# Custom environment wrapper to sample goals
class GoalEnvWrapper(gym.Wrapper):
    def __init__(self, env, goals):
        super().__init__(env)
        self.goals = goals
        self.current_goal = None
    
    def reset(self, **kwargs):
        self.current_goal = self.goals[np.random.randint(len(self.goals))]
        kwargs['options'] = {'goal': self.current_goal}
        obs, info = self.env.reset(**kwargs)
        return obs, info

# Create environment
def make_env(goals, seed=0):
    def _init():
        env = gym.make("FetchPush-v2", max_episode_steps=50)
        env = GoalEnvWrapper(env, goals)
        env = Monitor(env)
        return env
    return _init

# Create vectorized training environment
train_env = make_vec_env(make_env(train_goals), n_envs=1, seed=42)

# Initialize SAC model
model = SAC(
    policy="MultiInputPolicy",
    env=train_env,
    buffer_size=100000,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    verbose=1,
    seed=42
)

# Train the model
model.learn(total_timesteps=50000, progress_bar=True)

# Save the model
model.save("sac_fetchpush")

# Evaluate on training goals
mean_reward_train, std_reward_train = evaluate_policy(model, train_env, n_eval_episodes=20)
print(f"Mean reward on training goals: {mean_reward_train:.2f} +/- {std_reward_train:.2f}")

# Evaluate on test goals
def evaluate_on_specific_goals(model, env_name="FetchPush-v2", goals=TEST_GOALS, n_episodes=5):
    env = gym.make(env_name, max_episode_steps=50)
    rewards = []
    for goal in goals:
        for _ in range(n_episodes):
            obs, _ = env.reset(options={'goal': goal})
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

mean_reward_test, std_reward_test = evaluate_on_specific_goals(model)
print(f"Mean reward on test goals: {mean_reward_test:.2f} +/- {std_reward_test:.2f}")

# Compute generalization gap
generalization_gap = mean_reward_train - mean_reward_test
print(f"Generalization gap (train - test): {generalization_gap:.2f}")

# Clean up
train_env.close()
os.remove("sac_fetchpush.zip")