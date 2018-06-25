from agent_dir.agent import Agent
from colors import *
from tqdm import *
from collections import namedtuple

import scipy
import tensorflow as tf
import numpy as np
import os, sys
import random

SEED = 11037
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print(config)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


def prepro(o,image_size=[80,80]):
  """
  Call this function to preprocess RGB image to grayscale image if necessary
  This preprocessing code is from
      https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
  
  Input: 
  RGB image: np.array
      RGB screen of game, shape: (210, 160, 3)
  Default return: np.array 
      Grayscale image, shape: (80, 80, 1)
  
  """

  y = o.astype(np.uint8)

  resized = scipy.misc.imresize(y, image_size)
  return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
  def __init__(self, env, args):
    """
    Initialize every things you need here.
    For example: building your model
    """

    super(Agent_PG,self).__init__(env)
    self.args = args
    self.batch_size = args.batch_size
    self.lr = args.learning_rate
    self.action_dim = env.action_space.n # 6
    self.state_dim = env.observation_space.shape[0] # 210
    self.memory = []
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.step = 0


    self.ckpts_path = self.args.save_dir + "pg.ckpt"
    self.saver = tf.train.Saver(max_to_keep = 3)
    self.sess = tf.Session(config=config)

    if args.test_pg:
      #you can load your model here
      print('loading trained model')
      ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
      print(ckpt)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        self.step = self.sess.run(self.global_step)
        print('load step: ', self.step)
      else:
        print('load model failed! exit...')
        exit(0)
    else:
      self.init_model()

  def init_model(self):
    ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
    print(ckpt)
    if self.args.load_saver and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        self.step = self.sess.run(self.global_step)
        print('load step: ', self.step)
    else:
        print('Created new model parameters..')
        self.sess.run(tf.global_variables_initializer())

  def init_game_setting(self):
    """

    Testing function will call this function at the begining of new game
    Put anything you want to initialize if necessary

    """
    ##################
    # YOUR CODE HERE #
    ##################
    pass

  def storeTransition(self, s, action, reward, s_, done):
    tr = Transition(s, action, reward, s_, done)
    self.memory.append(tr)
    self.step = self.sess.run(self.add_global)
    # print(len(self.memory))

  def learn(self):
    pass


  def train(self):
    """
    Implement your training algorithm here
    """
    pbar = tqdm(range(self.args.episode_start, self.args.num_episodes))

    # 1, 5 : up; 3, 4: down; 2, 6: stop
    for episode in pbar:
      obs = self.env.reset()
      self.init_game_setting()
      episode_reward = 0.0
      for s in range(self.args.max_num_steps):
        action = self.make_action(obs, test=False)
        obs_, reward, done, info = self.env.step(action)
        episode_reward += reward
        self.storeTransition(obs, action, reward, obs_, done)
        obs = obs_
        if done:          
          break
      pbar.set_description("step: " + str(self.step) +  ", reward, " +  str(episode_reward))
      vt = self.learn()
      self.memory.clear()



  def make_action(self, observation, test=True):
    """
    Return predicted action of your agent

    Input:
        observation: np.array
            current RGB screen of game, shape: (210, 160, 3)

    Return:
        action: int
            the predicted action from trained model
    """
    ##################
    # YOUR CODE HERE #
    ##################
    return self.env.get_random_action()

