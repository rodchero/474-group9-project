import numpy as np
import gymnasium as gym
from collections import defaultdict
import coverage_gridworld 
import matplotlib.pyplot as plt

def get_action(state, q_table, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(0, 5)  # 5 actions (L, D, R, U, Stay)
    else:
        # Break ties randomly
        q_values = q_table[state]
        max_val = np.max(q_values)
        return np.random.choice(np.flatnonzero(q_values == max_val))

def train_sarsa(episodes=2000, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, decay_rate=0.995):
    # 1. INITIALIZE REWARDS HISTORY
    rewards_history = [] 
    
    env = gym.make("safe", render_mode=None, activate_game_status=False)
    q_table = defaultdict(lambda: np.zeros(5))
    epsilon = epsilon_start

    print("Training SARSA Agent...")
    for ep in range(episodes):
        obs, _ = env.reset()
        state = tuple(obs.tolist())
        
        action = get_action(state, q_table, epsilon)
        total_reward = 0
        done = False
        
        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = tuple(next_obs.tolist())
            
            next_action = get_action(next_state, q_table, epsilon)
            
            td_target = reward + gamma * q_table[next_state][next_action] * (not done)
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            
            state = next_state
            action = next_action
            total_reward += reward
            
        # 2. SAVE TOTAL REWARD FOR PLOTTING
        rewards_history.append(total_reward)
        
        epsilon = max(epsilon_min, epsilon * decay_rate)
        
        if (ep + 1) % 200 == 0:
            print(f"Episode {ep + 1}/{episodes} | Epsilon: {epsilon:.3f} | Last Reward: {total_reward:.1f}")

    env.close()
    
    return q_table

def test_agent(q_table, map_name="safe"):
    print(f"\nTesting Agent on map: {map_name}")
    env = gym.make(map_name, render_mode="human", activate_game_status=True)
    obs, _ = env.reset()
    state = tuple(obs.tolist())
    done = False
    
    while not done:
        # Fully greedy execution (epsilon = 0)
        action = get_action(state, q_table, epsilon=0)
        obs, reward, terminated, truncated, info = env.step(action)
        state = tuple(obs.tolist())
        done = terminated or truncated
        
    env.close()

if __name__ == "__main__":
    # 1. Train the agent
    trained_q_table = train_sarsa(episodes=2500)
    
    # 2. Test the agent visually
    test_agent(trained_q_table, map_name="sneaky_enemies")
    