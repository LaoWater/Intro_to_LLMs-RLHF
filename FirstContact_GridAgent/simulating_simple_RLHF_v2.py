import numpy as np
import random


# Define a simple grid environment
class SimpleGridEnv:
    def __init__(self, size=5):
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
            return self.state, 1, True  # Reward of 1, episode done
        else:
            return self.state, 0, False  # No reward yet, episode not done


# Function to map actions to movement directions
def map_action_to_movement(action):
    movement_map = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
    return movement_map.get(action, 'Unknown')


# Function to get human feedback based on individual axis movement (i, j) towards the goal
def get_human_feedback(current_state, previous_state, goal):
    print(f"Feedback enter Previous State: {previous_state}")
    print(f"Feedback enter Current State: {current_state}")
    # Track changes in x (row) and y (column) coordinates
    delta_x_current = abs(goal[0] - current_state[0])  # Absolute difference in x-axis (i) from goal
    delta_y_current = abs(goal[1] - current_state[1])  # Absolute difference in y-axis (j) from goal

    delta_x_previous = abs(goal[0] - previous_state[0])
    delta_y_previous = abs(goal[1] - previous_state[1])

    total_cur_difference = delta_x_current + delta_y_current
    total_prev_difference = delta_x_previous + delta_y_previous


    print(f"Total prev state distance from goal (X and Y axis combined) : {total_prev_difference}")
    print(f"Total current state distance from goal (X and Y axis combined) : {total_cur_difference}")

    if total_cur_difference > total_prev_difference:
        return "negative"
    elif total_cur_difference < total_prev_difference:
        return "positive"
    else:
        return "neutral"

# Define the agent with decaying exploration, policy updates, and boundary checking
class RLHF_Exploration_Agent:
    def __init__(self, initial_exploration_rate=0.5, decay=0.95, grid_size=5):
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.policy = {}
        self.exploration_rate = initial_exploration_rate  # Start with exploration
        self.decay = decay  # Decay rate for exploration
        self.grid_size = grid_size  # Size of the grid

    def choose_action(self, state):
        state_tuple = tuple(state)

        # Filter out actions that would lead outside the grid boundaries
        valid_actions = []
        if state[0] > 0:  # Can move Up
            valid_actions.append(0)
        if state[1] < self.grid_size - 1:  # Can move Right
            valid_actions.append(1)
        if state[0] < self.grid_size - 1:  # Can move Down
            valid_actions.append(2)
        if state[1] > 0:  # Can move Left
            valid_actions.append(3)

        # Explore: take a random valid action
        if random.random() < self.exploration_rate:
            return np.random.choice(valid_actions)

        # Exploit: use the learned policy if available, otherwise explore
        if state_tuple in self.policy:
            action = self.policy[state_tuple]
            if action in valid_actions:
                return action  # Only take the action if it's valid

        return np.random.choice(valid_actions)  # Fallback to a valid random action

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


# Simulate the RLHF process with decaying exploration and feedback
def rl_human_feedback_simulation(env, agent, steps=100):
    state = env.reset()
    state_history = []  # To track last few states for stuck detection
    stuck_counter = 0  # To detect if agent is stuck
    max_stuck_limit = 3  # Number of steps in a repeated state before forcing exploration

    for step in range(steps):
        # Choose action based on current state
        # print(f'Previous State: {previous_state}')
        previous_state = state[:]
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)  # Get next state after action, aah this updates the state.
        # but why? because immutable objects in python
        # print(f'Current State: {state}')

        # Print transition details
        # print(f'Chosen AI Action: {map_action_to_movement(action)}')
        # print(f'Next State: {next_state}')

        # Simulate human feedback based on previous and current state distances to goal
        feedback = get_human_feedback(next_state, previous_state, env.goal)
        print(
            f"Step: {step + 1}, Action: {map_action_to_movement(action)}, Feedback: {feedback}, Exploration Rate: {agent.exploration_rate:.2f} \n")

        # Keep track of recent states to detect if agent is stuck
        state_history.append(state[:])
        if len(state_history) > 5:  # Keep the history size limited to 5
            state_history.pop(0)

        # Detect if agent is stuck in a loop by checking if it repeats recent states
        if state in state_history[:-1]:  # Check if current state has been visited recently
            stuck_counter += 1
        else:
            stuck_counter = 0  # Reset counter if progress is made

        # If stuck for too long, force exploration to avoid loops
        if stuck_counter >= max_stuck_limit:
            print("Agent is stuck. Forcing exploration.")
            agent.exploration_rate = 1.0  # Reset exploration to force new actions
            stuck_counter = 0  # Reset the counter after forcing exploration


        # Update previous state for next step comparison
        previous_state = state[:]

        # Agent updates policy based on feedback
        agent.update_policy(state, action, feedback)


        # print(f'Previous state: {previous_state}')
        state = next_state[:]  # Ensure state is updated to next state for the next iteration

        if done:
            print("Goal reached!")
            break


# Main function to run the simulation
if __name__ == "__main__":
    env = SimpleGridEnv()
    agent = RLHF_Exploration_Agent(initial_exploration_rate=0.5,
                                   decay=0.9)  # 50% exploration rate initially, decaying by 10% each step
    rl_human_feedback_simulation(env, agent)
