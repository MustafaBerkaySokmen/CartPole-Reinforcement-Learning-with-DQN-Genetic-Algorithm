import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Updated memory size
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.5   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001  # Updated learning rate
        self.model = self._build_model()

    def _build_model(self):
        # Enhanced model with additional layers and dropout
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.1))  # Dropout layer to prevent overfitting
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        # Convert state to numpy array for prediction
        state = np.array([state])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def evaluate_agent(agent, num_episodes=50):
    env = gym.make('CartPole-v1')
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        while True:
            action = agent.act(state)
            step_result = env.step(action)
            next_state = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            total_reward += reward
            state = next_state
            if done:
                break
    env.close()
    return total_reward / num_episodes

def render_agent_performance(agent):
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset()
    done = False
    time_survived = 0
    while not done:
        env.render()
        action = agent.act(state)
        step_result = env.step(action)
        next_state = step_result[0]
        done = step_result[2]
        state = next_state
        time_survived += 1
    print(f"Time Survived: {time_survived}")
    env.close()

def genetic_algorithm(agents, generations=100, top_k=10, render_best=False):
    for g in range(generations):
        fitness_scores = [evaluate_agent(agent, num_episodes=50) for agent in agents]
        top_agents = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:top_k]

        if render_best:
            best_agent = agents[top_agents[0]]
            render_agent_performance(best_agent)

        new_agents = []
        for _ in range(len(agents) - top_k):
            parents = np.random.choice(top_agents, 2, replace=False)
            child = crossover(agents[parents[0]], agents[parents[1]])
            mutate(child)
            new_agents.append(child)

        for i, idx in enumerate(sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:len(new_agents)]):
            agents[idx] = new_agents[i]
        print(f"Generation {g} complete.")

def crossover(agent1, agent2):
    child = DQNAgent(agent1.state_size, agent1.action_size)
    weights1 = agent1.model.get_weights()
    weights2 = agent2.model.get_weights()
    child_weights = [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]
    child.model.set_weights(child_weights)
    return child

def mutate(agent):
    mutated_weights = []
    for weights in agent.model.get_weights():
        mutation = np.random.normal(loc=0.0, scale=0.1, size=weights.shape)
        mutated_weights.append(weights + mutation)
    agent.model.set_weights(mutated_weights)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agents = [DQNAgent(state_size, action_size) for _ in range(1000)]

    genetic_algorithm(agents, generations=20, render_best=True)
