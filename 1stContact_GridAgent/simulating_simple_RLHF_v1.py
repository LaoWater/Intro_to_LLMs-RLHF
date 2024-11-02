import random

import numpy as np


# small RLHF simulation using basic Python code. Let’s start by defining a simple reinforcement learning agent that
# receives human feedback on how to navigate a basic environment (such as navigating a grid).
# Step 1: Environment Setup
# Define a simple grid environment

class SimpleGridEnv:
    def __init__(self, size=3):
        self.size = size
        self.state = [0, 0]  # Start in top-left corner
        self.goal = [size - 1, size - 1]  # Goal is bottom-right corner

    def reset(self):
        self.state = [0, 0]
        return self.state

    def step(self, action):
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1
        elif action == 1 and self.state[1] < self.size - 1:
            self.state[1] += 1
        elif action == 2 and self.state[0] < self.size - 1:
            self.state[0] += 1
        elif action == 3 and self.state[1] > 0:
            self.state[1] -= 1

        # If agent reaches the goal, they get a reward
        if self.state == self.goal:
            return self.state, 1, True  # Reward of 1
        else:
            return self.state, 0, False  # No reward yet


# Step 2: RL Agent with Human Feedback
# Now, let's define a basic agent that uses human feedback to update its behavior. The human feedback will be simulated
# as ranking which action is the best for reaching the goal.)
# Define the agent with decaying exploration and policy updates based on feedback
class RLHF_Exploration_Agent:
    def __init__(self, initial_exploration_rate=0.5, decay=0.95):
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.policy = {}
        self.exploration_rate = initial_exploration_rate  # Start with exploration
        self.decay = decay  # Decay rate for exploration

    def choose_action(self, state):
        state_tuple = tuple(state)
        # Explore: take a random action
        if random.random() < self.exploration_rate:
            return np.random.choice(self.actions)
        # Exploit: use the learned policy
        elif state_tuple in self.policy:
            return self.policy[state_tuple]
        return np.random.choice(self.actions)

    def update_policy(self, state, action, feedback):
        state_tuple = tuple(state)
        if feedback == "positive" or feedback == "neutral":
            self.policy[state_tuple] = action  # Keep the action if it's positive/neutral
        else:
            # Try new action if feedback is negative (explore)
            new_action = np.random.choice([a for a in self.actions if a != action])
            self.policy[state_tuple] = new_action

        # Decay the exploration rate over time to exploit more as agent learns
        self.exploration_rate *= self.decay
        self.exploration_rate = max(self.exploration_rate, 0.1)  # Keep exploration > 10%


# Simulate human feedback based on proximity to the goal
def get_human_feedback(state, goal):
    if state == goal:
        return "positive"
    distance = abs(state[0] - goal[0]) + abs(state[1] - goal[1])

    # Provide nuanced feedback
    if distance == 1:  # Almost at the goal
        return "neutral"  # Close enough
    elif distance <= 2:  # Moving towards goal
        return "positive"
    else:
        return "negative"


# Simulate the RLHF process with decaying exploration and feedback
def rl_human_feedback_simulation(env, agent, steps=30):
    state = env.reset()
    for step in range(steps):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        # Simulate human feedback based on proximity to goal
        feedback = get_human_feedback(next_state, env.goal)
        print(
            f"Step: {step + 1}, State: {state}, Action: {action}, Feedback: {feedback}, Exploration Rate: {agent.exploration_rate:.2f}")

        # Agent updates policy based on feedback
        agent.update_policy(state, action, feedback)

        state = next_state
        if done:
            print("Goal reached!")
            break


# Main function to run the simulation
if __name__ == "__main__":
    env = SimpleGridEnv()
    agent = RLHF_Exploration_Agent(initial_exploration_rate=0.5,
                                   decay=0.9)  # 50% exploration rate initially, decaying by 10% each step
    rl_human_feedback_simulation(env, agent)