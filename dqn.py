import numpy as np
import pandas as pd
import tensorflow as tf
import win32api, win32con
import time
from collections import deque
from score_ai import ScoreKeeper
from screen import screenGrab

ACTION_SPACE = 3
MODEL_DIR = './models/'
SCORE_MODEL = MODEL_DIR + 'scoreModel2'

# Configuration values
class config:
    num_episodes = 50
    epsilon = 1
    epsilon_discount = 0.95
    batch_size=32
    discount = 0.99

    NUM_USER_TRAIN = 10

    optimizer = tf.keras.optimizers.Adam(1e-4)
    mse = tf.keras.losses.MeanSquaredError()

# The neural network model responsible for predicting next action
class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1, 899, 1918, 1))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPool2D((2, 2))
        self.reshape = tf.keras.layers.Reshape((1675520,), input_shape=((1, 110, 238, 64)))
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(ACTION_SPACE, activation='softmax')

    def call(self, input):
        
        x = self.conv1(input)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.pool(x)

        x = self.dense1(x)
        x = self.reshape(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def train(self, target_nn, env, state, action, reward, next_state, done):
        '''Perform training on batch of data from replay buffer'''
        # Calculate targets
        next_qs = target_nn.get_prediction(next_state)

        max_next_qs = tf.reduce_max(next_qs, axis=-1)

        target = reward + 1 * config.discount * max_next_qs
        #target = reward + (1. - done) * config.discount * max_next_qs

        with tf.GradientTape() as tape:
            qs = self.get_prediction(state)
            # Create one hot
            action_mask = tf.zeros((ACTION_SPACE)).numpy()
            action_mask[action] = 1
            masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
            loss = config.mse(target, masked_qs)
        grads = tape.gradient(loss, self.trainable_variables)
        config.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def get_prediction(self, input):
        x = np.reshape(input, (1, 899, 1918, 1))
        x = self.call(x)
        return x


# main class that brings everything together
class Trainer:
    def __init__(self):
        self.env = Environment()
        self.buffer = ReplayBuffer(100000)
        self.cur_frame = 0
        self.target_nn = DQN()
        self.main_nn = DQN()

    # picks random action at the beginning
    # as training continues will get action from NN
    def select_epsilon_greedy_action(self, state):
        # epsilon is probability of random action other wise use NN predicted action
        random = tf.random.uniform((1,))
        if random < config.epsilon:
            return self.env.random_action()
        else:
            output = self.main_nn.get_prediction(state)
            action = np.argmax(output)
            return action

    def watch_action(self):
        while True:
            if win32api.GetAsyncKeyState(win32con.VK_LEFT):
                print('Left')
                return 0
            if win32api.GetAsyncKeyState(win32con.VK_RIGHT):
                print('right')
                return 1
            if win32api.GetAsyncKeyState(win32con.VK_SPACE):
                print('space')
                return 2


    def watch(self):
        state = self.env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = self.watch_action()
            next_state, reward, done = self.env.observe()

            # save to experience replay
            self.buffer.add(state, action, reward, next_state, done)
            if reward > ep_reward:
                ep_reward = reward

            # prepare for next frame
            self.cur_frame += 1
            state = next_state

            # copy main_nn weights to target_nn weights
            if self.cur_frame % 2000 == 0:
                self.target_nn.set_weights(self.main_nn.get_weights())

        return ep_reward

    def handsOn(self):
        state = self.env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = self.select_epsilon_greedy_action(state)
            next_state, reward, done = self.env.step(action)

            # save to experience replay
            self.buffer.add(state, action, reward, next_state, done)
            if reward > ep_reward:
                ep_reward = reward

            # prepare for next frame
            self.cur_frame += 1
            state = next_state

            # copy main_nn weights to target_nn weights
            if self.cur_frame % 2000 == 0:
                self.target_nn.set_weights(self.main_nn.get_weights())

        return ep_reward

            


    def train(self):
        last_100_ep_rewards = []
        # each episode is a game
        for episode in range(0, config.num_episodes):
            ep_reward = 0

            if episode < config.NUM_USER_TRAIN:
                ep_reward = self.watch()
            else:
                ep_reward = self.handsOn()

            # Train main_nn
            if len(self.buffer) >= config.batch_size:
                state, action, reward, next_state, done = self.buffer.sample()

                print('Training...')
                self.main_nn.train(self.target_nn, self.env, state, action, reward, next_state, done)

            # prepare for next episode and print results
            if episode < config.num_episodes * config.epsilon_discount:
                config.epsilon -= config.epsilon_discount / config.num_episodes

            # if num of rewards has reached 100 remove first one
            if len(last_100_ep_rewards) == 100:
                last_100_ep_rewards = last_100_ep_rewards[1:]

            last_100_ep_rewards.append(ep_reward)

            # print current results
            if episode % 1 == 0:
                print(f'Episode {episode}/{config.num_episodes}. Epsilon: {config.epsilon:.3f}. \n'
				f'Average Reward: {np.mean(last_100_ep_rewards):.2f}\n')
                self.target_nn.save_weights(filepath=MODEL_DIR + 'slither_model_ep_' + str(episode) + '_reward_' +str(ep_reward))

        return self.target_nn

        


# this class interacts with the environment
class Environment:
    from screen import screenGrab
    import win32api, win32con

    def __init__(self):
        self.action_space = [Action('left', self.__press_left), 
                            Action('right', self.__press_right), 
                            Action('space', self.__press_space)]
        self.score_ai = ScoreKeeper()
        self.score_ai.load_weights(SCORE_MODEL).expect_partial()

        self.last_3_rewards = []
        

    # Actions
    def __press_left(self):
        win32api.keybd_event(0x25, 0,0,0)
        time.sleep(.1)
        win32api.keybd_event(0x25,0 , win32con.KEYEVENTF_KEYUP ,0)

    def __press_right(self):
        win32api.keybd_event(0x27, 0,0,0)
        time.sleep(.1)
        win32api.keybd_event(0x27,0 , win32con.KEYEVENTF_KEYUP ,0)

    def __press_space(self):
        win32api.keybd_event(0x20, 0,0,0)
        time.sleep(.1)
        win32api.keybd_event(0x20,0 , win32con.KEYEVENTF_KEYUP ,0)

    def reset(self):
        # click play button
        #win32api.SetCursorPos((800, 600))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
        time.sleep(.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
        time.sleep(3)

        # get state
        screen = screenGrab()
        return screen[2]

    def observe(self):
        screen = screenGrab()
        game_screen = screen[2]
        reward = self.score_ai.getScore(screen[1])

        if len(self.last_3_rewards) == 5:
                self.last_3_rewards = self.last_3_rewards[1:]

        self.last_3_rewards.append(reward)
        avg_reward = np.mean(self.last_3_rewards)

        done = False
        if reward == 0:
            done = True

        print('Reward: ' + str(avg_reward))
        return (game_screen, avg_reward, done)

    def step(self, action):
        self.action_space[action].perform()
        state, reward, done = self.observe()
        return state, reward, done

    def random_action(self):
        action = np.random.randint(0, len(self.action_space))
        return action



# Will perform action function
class Action:
    def __init__(self, name, action):
        self.perform = action
        self.name = name


# Allows replaying of previous environments
class ReplayBuffer(object):
    def __init__(self, size):
        self.buffer = deque(maxlen = size)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def __len__(self):
        return len(self.buffer)
        
    def sample(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer))
        # currently just returns 1
        elem = self.buffer[idx]
        state, action, reward, next_state, done = elem

        return state, action, reward, next_state, done