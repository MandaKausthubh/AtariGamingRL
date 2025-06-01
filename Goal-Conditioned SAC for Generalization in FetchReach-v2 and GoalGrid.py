import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os
from gymnasium import spaces

# Set random seed for reproducibility
np.random.seed(42)

# Custom GoalGrid environment
class GoalGrid(gym.Env):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.float32),
            'desired_goal': spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(4)  # Up, down, left, right
        self.position = None
        self.goal = None
        self.max_steps = 50
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([0, 0], dtype=np.float32)
        self.goal = options.get('goal', np.array([self.size-1, self.size-1], dtype=np.float32))
        self.current_step = 0
        obs = {
            'observation': self.position.copy(),
            'achieved_goal': self.position.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs, {}

    def step(self, action):
        self.current_step += 1
        new_pos = self.position.copy()
        if action == 0:  # Up
            new_pos[1] = min(new_pos[1] + 1, self.size-1)
        elif action == 1:  # Down
            new_pos[1] = max(new_pos[1] - 1, 0)
        elif action == 2:  # Left
            new_pos[0] = max(new_pos[0] - 1, 0)
        elif action == 3:  # Right
            new_pos[0] = min(new_pos[0] + 1, self.size-1)
        self.position = new_pos

        obs = {
            'observation': self.position.copy(),
            'achieved_goal': self.position.copy(),
            'desired_goal': self.goal.copy()
        }
        reward = -np.linalg.norm(self.position - self.goal)
        terminated = np.array_equal(self.position, self.goal)
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass  # Optional: Add rendering if needed

# Register custom environment
gym.register(id='GoalGrid-v0', entry_point='__main__:GoalGrid')

# Experiment function
def run_experiment(env_name, train_goals, test_goals, total_timesteps=50000, n_eval_episodes=20):
    # Create environment
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

    def make_env(goals, seed=0):
        def _init():
            env = gym.make(env_name, max_episode_steps=50)
            env = GoalEnvWrapper(env, goals)
            env = Monitor(env)
            return env
        return _init

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

    # Train
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save model
    model_path = f"sac_{env_name.lower().replace('-', '_')}"
    model.save(model_path)

    # Evaluate on training goals
    mean_reward_train, std_reward_train = evaluate_policy(model, train_env, n_eval_episodes=n_eval_episodes)
    print(f"{env_name} - Mean reward on training goals: {mean_reward_train:.2f} +/- {std_reward_train:.2f}")

    # Evaluate on test goals
    def evaluate_on_specific_goals(model, env_name, goals, n_episodes=5):
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

    mean_reward_test, std_reward_test = evaluate_on_specific_goals(model, env_name, test_goals)
    print(f"{env_name} - Mean reward on test goals: {mean_reward_test:.2f} +/- {std_reward_test:.2f}")

    # Compute generalization gap
    generalization_gap = mean_reward_train - mean_reward_test
    print(f"{env_name} - Generalization gap (train - test): {generalization_gap:.2f}")

    # Clean up
    train_env.close()
    os.remove(f"{model_path}.zip")

# Define goals for FetchReach-v2
FETCH_TRAIN_GOAL_RANGE = [-0.1, 0.1]  # Training goals in a cube
FETCH_TEST_GOALS = [np.array([0.15, 0.15, 0.15]), np.array([0.2, 0.2, 0.2])]  # Outside training range
FETCH_NUM_TRAIN_GOALS = 100
fetch_train_goals = [np.random.uniform(*FETCH_TRAIN_GOAL_RANGE, size=3) for _ in range(FETCH_NUM_TRAIN_GOALS)]

# Define goals for GoalGrid
GRID_TRAIN_GOAL_RANGE = [0, 2]  # Training goals in top-left quadrant
GRID_TEST_GOALS = [np.array([3, 3], dtype=np.float32), np.array([4, 4], dtype=np.float32)]  # Bottom-right quadrant
GRID_NUM_TRAIN_GOALS = 50
grid_train_goals = [np.array([np.random.randint(*GRID_TRAIN_GOAL_RANGE), np.random.randint(*GRID_TRAIN_GOAL_RANGE)], dtype=np.float32) for _ in range(GRID_NUM_TRAIN_GOALS)]

# Run experiments
run_experiment("FetchReach-v2", fetch_train_goals, FETCH_TEST_GOALS)
run_experiment("GoalGrid-v0", grid_train_goals, GRID_TEST_GOALS)