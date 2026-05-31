import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
import random
from collections import deque
import matplotlib.pyplot as plt

STUDENT_ID = 'a1818124'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = deque(maxlen=250)  # a data structure of your choice (D in the Algorithm 2)
        # self.storage2 = deque(maxlen=150)  # a data structure of your choice (D in the Algorithm 2)
        # A neural network MLP model which can be used as Q
        self.network = MLPRegression(input_dim = 5, output_dim = 2, learning_rate=0.0001)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim = 5, output_dim = 2, learning_rate=0.0001)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 0  # probability ε in Algorithm 2
        self.epsilon_decay = 0.999  # Decay rate for epsilon
        self.epsilon_min = 0.01  # Minimum epsilon
        self.n = 32  # the number of samples you'd want to draw from the storage each time
        self.discount_factor = 0.99  # γ in Algorithm 2
        self.prev_score = 0
        self.next_pipe = None
        self.next_pipe2 = None
        self.next_pipe3 = None
        self.epsilons = []
        self.epsilons_counter = 0
        # self.ispipe = 0
        self.done_rewards = []

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def build_state(self, state: dict) -> np.ndarray:
        screen_height = state['screen_height']

        # bird_y = (state['bird_y']+15) / (screen_height+62)  # Normalize bird's vertical position (0-1)
        bird_velocity = (state['bird_velocity'] + 8) / 32  # Normalize bird's velocity (0-1)

        pipe_count = len(state['pipes'])
        if pipe_count > 0:
            # if self.ispipe == 0:
            #     print("<< Pipe Appears >>")
            #     self.ispipe += 1

            if state['bird_x'] < state['pipes'][0]['x'] + state['pipes'][0]['width']:
                self.next_pipe = state['pipes'][0]
                if pipe_count > 1:
                    self.next_pipe2 = state['pipes'][1]
                    if pipe_count > 2:
                        self.next_pipe3 = state['pipes'][2]
            else:
                # print("< Pipe Passed! >") 
                self.next_pipe = state['pipes'][1]
                if pipe_count > 2:
                    self.next_pipe2 = state['pipes'][2]
                    if pipe_count > 3:
                        self.next_pipe3 = state['pipes'][3]
            x_distance = (self.next_pipe['x'] + self.next_pipe['width'] - state['bird_x']) / (state['screen_width'] - state['bird_x'] + self.next_pipe['width'])  # Normalize horizontal distance
            y_distance = (state['bird_y'] + state['bird_height']/2 - (self.next_pipe['top'] + self.next_pipe['bottom'])/2 + screen_height) / (2 * screen_height)  # Normalize vertical distance
            if self.next_pipe2: y_distance2 = (state['bird_y'] + state['bird_height']/2 - (self.next_pipe2['top'] + self.next_pipe2['bottom'])/2 + screen_height) / (2 * screen_height)  # Normalize vertical distance
            else: y_distance2 = y_distance
            if self.next_pipe3: y_distance3 = (state['bird_y'] + state['bird_height']/2 - (self.next_pipe3['top'] + self.next_pipe3['bottom'])/2 + screen_height) / (2 * screen_height)
            else: y_distance3 = y_distance2
        else:
            # if self.ispipe > 0:
            #     print("!!! Pipe Error: self.ispipe (not 0):", self.ispipe, "| pipe count:", pipe_count)
            x_distance = 1
            y_distance = (state['bird_y'] + state['bird_height']/2) / screen_height / 1.05
            y_distance2 = y_distance
            y_distance3 = y_distance2

        phi = np.array([bird_velocity, x_distance, y_distance, y_distance2, y_distance3])
        for i in range(len(phi)):
            if phi[i] > 1 or phi[i] < 0:
                print("!!! Normalization Error - Feature", i, ":", phi[i], "| mileage:", state['mileage'])
        return phi

    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        This function should be called when the agent action is requested.
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """
        phi_t = self.build_state(state)
        if self.mode == 'train':
            if np.random.rand() < self.epsilon:
                a_t = np.random.choice([0, 1])
            else:
                q_values = self.network.predict(phi_t)
                a_t = np.argmax(q_values)
            # Store the transition in memory
            self.storage.append((phi_t, a_t, None, None)) 
        else:
            q_values = self.network.predict(phi_t)
            a_t = np.argmax(q_values)
        return a_t

    def reward(self, state: dict, phi: dict) -> float:
        r_centre = 0
        if self.next_pipe:
            if state['bird_y'] < self.next_pipe['top']:
                r_centre = -(self.next_pipe['top'] - state['bird_y']) / state['screen_height']
            elif state['bird_y'] + state['bird_height'] > self.next_pipe['bottom']:
                r_centre = -(state['bird_y'] + state['bird_height'] - self.next_pipe['bottom']) / state['screen_height'] * 0.99
            elif self.next_pipe2:
                if state['bird_y'] < self.next_pipe2['top']:
                    r_centre = -(self.next_pipe2['top'] - state['bird_y']) / state['screen_height'] * 0.5
                elif state['bird_y'] + state['bird_height'] > self.next_pipe2['bottom']:
                    r_centre = -(state['bird_y'] + state['bird_height'] - self.next_pipe2['bottom']) / state['screen_height'] * 0.5 * 0.99
                elif self.next_pipe3:
                    if state['bird_y'] < self.next_pipe3['top']:
                        r_centre = -(self.next_pipe3['top'] - state['bird_y']) / state['screen_height'] * 0.25
                    elif state['bird_y'] + state['bird_height'] > self.next_pipe3['bottom']:
                        r_centre = -(state['bird_y'] + state['bird_height'] - self.next_pipe3['bottom']) / state['screen_height'] * 0.25 * 0.99
        else:  r_centre = -abs(phi[2] - 0.5)/10

        # r_centre = -abs(phi[2] - 0.5)/5 - abs(phi[3] - 0.5)/10
        # r_centre = 0
        r_mileage = state['mileage'] * 0.0005
        # r_mileage = state['mileage']/1047

        if state['done']:
            self.prev_score = 0
            # self.ispipe = 0
            # print("done_type:", state['done_type'])
            if state['done_type'] == 'well_done': r_total = 1
            elif state['done_type'] == 'hit_pipe': r_total = -0.3 + r_centre + r_mileage
            elif state['done_type'] == 'offscreen': r_total = -0.35 + r_centre + r_mileage
            else: 
                print("!!! state['done'] Error - state['done_type']:", state['done_type'])
                r_total = -0.5 + r_centre + r_mileage
            # print("<< DEAD >>:", r_total,"(reward) =", state['done_type'], "+", r_centre, "(r_centre) +", r_mileage, "(r_mileage)")
            self.done_rewards.append(r_total)
            
        elif state['score'] > self.prev_score:
            self.prev_score = state['score']

            r_total = 0.2 * self.prev_score + r_centre + r_mileage
            # print("<< Score:", self.prev_score, ">>")

        else: r_total = r_centre + r_mileage

        if r_total < -1 or r_total > 1:
            print("!!! Normalization Error:", r_total,"(reward) =", state['done_type'], "+", r_centre, "(r_centre) +", r_mileage, "(r_mileage)")
        return r_total

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        # following pseudocode to implement this function
        if self.mode == 'train':
            phi_t1 = self.build_state(state)
            r_t = self.reward(state, phi_t1)
            if state['done']: q_t1 = 0
            else: q_t1 = np.max(self.network2.predict(phi_t1))

            # Update the last transition in memory
            if self.storage:
                last_transition = self.storage.pop()
                self.storage.append((last_transition[0], last_transition[1], r_t, q_t1))

            X, Y, W = [], [], []
            # all_storage = list(self.storage) + list(self.storage2)
            # batch_size = min(self.n, len(all_storage))
            # minibatch = random.sample(all_storage, batch_size)
            batch_size = min(self.n, len(self.storage))
            minibatch = random.sample(self.storage, batch_size)
            for transition in minibatch:
                phi_j, a_j, r_j, q_j1 = transition
                w_j = np.zeros(2)
                w_j[a_j] = 1

                # y_j = self.network.predict(phi_j).copy()
                y_j = np.zeros(2)
                if q_j1 is None:
                    print("!!! Weird Error - receive_after_action_observation: q_j1 is None")
                    y_j[a_j] = r_j
                else:
                    y_j[a_j] = r_j + self.discount_factor * q_j1
                # y_j = r_j + self.discount_factor * q_j1
                X.append(phi_j)
                Y.append(y_j)
                W.append(w_j)
            # print(np.array(Y).shape)
            self.network.fit_step(np.array(X), np.array(Y), np.array(W))

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            if (self.epsilons_counter == 100):
                self.epsilons.append(self.epsilon)
                self.epsilons_counter = 0
            else:
                self.epsilons_counter += 1
    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())

import os
def save_to_txt(epsilon, filename="epsilon.txt"):
    try:
        with open(filename, "w") as f:
            f.write(str(epsilon))
    except Exception as e:
        print(f"!!! Error saving epsilon to {filename}: {e}")

def load_from_txt(filename="epsilon.txt", default_epsilon=1.0):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                epsilon = float(f.read().strip())
            return epsilon
        except Exception as e:
            print(f"!!! Error loading epsilon from {filename}: {e}. Using default.")
            return default_epsilon
    else:
        print(f"!!! {filename} not found. Using default epsilon: {default_epsilon}")
        return default_epsilon

def ave_last_n(arr, n):
    if len(arr) < n:
        return sum(arr) / len(arr)
    return sum(arr[-n:]) / n

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)

    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level, game_length=10)
    # agent = MyAgent(show_screen=False)
    agent = MyAgent(show_screen=False, load_model_path='my_model.ckpt')
    # agent = MyAgent(show_screen=False)
    
    
    # agent.epsilon = load_from_txt()

    episodes = 10000
    max_mileage = load_from_txt("mileage.txt")
    train_scores, train_mileages = [], []
    
    for episode in range(episodes):
        env.play(player=agent)

        # env.score has the score value from the last play
        # env.mileage has the mileage value from the last play
        # print(env.score)
        # print(env.mileage)

        train_scores.append(env.score)
        train_mileages.append(env.mileage)
        print('episode', episode, ':', env.score, env.mileage)

        # store the best model based on your judgement
        ave_mileages = ave_last_n(train_mileages, 5)
        if  ave_mileages > max_mileage:
            max_mileage = ave_mileages
            # if (env.mileage > 500): agent.storage2 = agent.storage
            print('[Model Saved]')
            agent.save_model(path='my_model.ckpt')
        # if episode % 500 == 0:
        #     agent.epsilon = 0.5

        # you'd want to clear the memory after one or a few episodes
        # agent.storage.clear()

        # you'd want to update the fixed Q-target network (Q_f) with Q's model parameter after one or a few episodes
        MyAgent.update_network_model(net_to_update=agent.network2, net_as_source=agent.network)
    
    save_to_txt(max_mileage, filename="mileage.txt")
    save_to_txt(agent.epsilon)
    # agent.save_model(path='my_model.ckpt')

    # Create the x-axis (episodes)
    x_axis = range(episodes)
    # Create the figure and subplots
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    # Plot 1: Training Scores
    plt.subplot(3, 2, 1)  # 1 row, 2 columns, first plot
    plt.plot(x_axis, train_scores, label='Training Score')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Training Score of each Episode')
    plt.legend()
    plt.grid(True)
    # Plot 2: Training Mileages
    plt.subplot(3, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(x_axis, train_mileages, label='Training Mileage')
    plt.xlabel('Episodes')
    plt.ylabel('Mileage')
    plt.title('Training Mileage of each Episode')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)  # 1 row, 2 columns, second plot
    plt.plot(range(len(agent.epsilons)), agent.epsilons, label='Training Epsilons', marker='o', linewidth=0.5, markersize=0.5)
    plt.xlabel('Decays')
    plt.ylabel('Epsilons')
    plt.title('Training Epsilon vs. Decays')
    plt.legend()
    plt.grid(True)

    rewards = [transition[2] for transition in agent.storage]
    plt.subplot(3, 2, 4)  # 1 row, 2 columns, second plot
    plt.scatter(range(len(agent.done_rewards)), agent.done_rewards, label="Done Rewards", s=10, c='green')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Finishing rewards of each episode')
    plt.legend()
    plt.grid(True)

    rewards = [transition[2] for transition in agent.storage]
    plt.subplot(3, 2, 5)  # 1 row, 2 columns, second plot
    plt.scatter(range(len(rewards)), rewards, label="Last Rewards", s=10)
    plt.xlabel('Actions')
    plt.ylabel('Reward')
    plt.title('last 250 rewards (in storage)')
    plt.legend()
    plt.grid(True)

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    # Show the plot
    plt.show()

    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=True, load_model_path='my_model.ckpt', mode='eval')

    print("=== EVAL START ===")
    episodes = 3
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
    print(np.median(scores))

    
