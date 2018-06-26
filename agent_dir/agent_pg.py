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

# 1, 5 : up; 3, 4: down; 2, 6: stop
actions_in = {'up': 2, 'down': 3}
actions_out = {actions_in['up']: 1, actions_in['down']: 0}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))


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
  y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
  y = y.astype(np.uint8)
  resized = scipy.misc.imresize(y, image_size)
  return np.expand_dims(resized.astype(np.float32), axis=2)


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
    self.gamma = args.gamma
    self.hidden_dim = 256
    self.action_dim = env.action_space.n # 6
    self.state_dim = env.observation_space.shape[0] # 210
    self.memory = []
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.step = 0
    self.s = tf.placeholder(tf.float32, [None, 6400], name='s')
    self.act = tf.placeholder(tf.float32, [None, 1], name='act')
    self.advantage =  tf.placeholder(
            tf.float32, [None, 1], name='advantage')
    
    self.build_net()
    self.buildOptimizer()

    self.ckpts_path = self.args.save_dir + "pg.ckpt"
    self.saver = tf.train.Saver(max_to_keep = 3)
    self.sess = tf.Session(config=config)
    self.writer = tf.summary.FileWriter(self.args.log_dir, graph=self.sess.graph)

    if args.test_pg:
      #you can load your model here
      print('loading trained model')
      ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
      print('ckpt', ckpt)
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
    print('ckpt: ', ckpt)
    if self.args.load_saver and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        self.step = self.sess.run(self.global_step)
        print('load step: ', self.step)
    else:
        print('Created new model parameters..')
        self.sess.run(tf.global_variables_initializer())

  def build_net(self):
    with tf.variable_scope('fc1'):
      W_fc1 = self.init_W(shape=[6400, self.hidden_dim])
      b_fc1 = self.init_b(shape=[self.hidden_dim])
      fc1 = tf.nn.bias_add(tf.matmul(self.s, W_fc1), b_fc1)
      h_fc1 = tf.nn.relu(fc1)
    with tf.variable_scope('fc2'):
      W_fc2 = self.init_W(shape=[self.hidden_dim, 1])
      b_fc2 = self.init_b(shape=[1])
      fc2 = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2)
      self.up_prob = tf.nn.sigmoid(fc2)

  def buildOptimizer(self):

    self.loss = tf.losses.log_loss(labels=self.act, predictions=self.up_prob, weights=self.advantage)
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


  def init_W(self, shape, name='weights', 
    w_initializer=tf.contrib.layers.xavier_initializer()):

    return tf.get_variable(
      name=name,
      shape=shape, 
      initializer=w_initializer)

  def init_b(self, shape, name='biases', 
    b_initializer=tf.contrib.layers.xavier_initializer()):

    return tf.get_variable(
      name=name,
      shape=shape,
      initializer=b_initializer)


  def init_game_setting(self):
    """

    Testing function will call this function at the begining of new game
    Put anything you want to initialize if necessary

    """
    ##################
    # YOUR CODE HERE #
    ##################
    pass

  def storeTransition(self, s, action, reward):
    tr = Transition(s, action, reward)
    self.memory.append(tr)
    self.step = self.sess.run(self.add_global)
    # print(len(self.memory))

  def learn(self):

    state_batch = [mem.state for mem in self.memory]
    state_batch = np.array(state_batch).reshape(-1, 6400)
    action_batch = [mem.action for mem in self.memory]
    action_batch = np.array(action_batch).reshape(-1, 1)

    reward_batch = self.discount_and_norm_rewards()
    # reward_batch = [mem.reward for mem in self.memory]
    reward_batch = np.array(reward_batch).reshape(-1, 1)

    self.sess.run(self.train_op, feed_dict={
        self.s: state_batch,
        self.act: action_batch,
        self.advantage: reward_batch
      })

  def discount_and_norm_rewards(self):
    reward_batch = [mem.reward for mem in self.memory]
    discounted = np.zeros_like(reward_batch)
    # discount episode rewards
    for t in range(len(reward_batch)):
      discounted_sum = 0
      discount = 1
      for k in range(t, len(reward_batch)):
        discounted_sum += reward_batch[k] * discount
        discount *= self.gamma
        if reward_batch[k] != 0:
          break
      discounted[t] = discounted_sum
    # normalize episode rewards
    discounted -= np.mean(discounted)
    discounted /= np.std(discounted)

    return discounted

  def train(self):
    """
    Implement your training algorithm here
    """
    pbar = tqdm(range(self.args.episode_start, self.args.num_episodes))
    avg_reward = []
    total_mem_len = 0.0
    for episode in pbar:
      self.init_game_setting()
      obs = self.env.reset()
      obs = prepro(obs)
      action = self.env.action_space.sample()
      obs_, _, _, _ = self.env.step(action)
      obs_ = prepro(obs_)
      episode_reward = 0.0

      for s in range(self.args.max_num_steps):
        #self.env.env.render()
        state = obs_ - obs
        obs = obs_

        action = self.make_action(state, test=False)
        obs_, reward, done, _ = self.env.step(action)
        obs_ = prepro(obs_)
        episode_reward += reward
        self.storeTransition(state, actions_out[action], reward)

        if self.step % self.args.saver_steps == 0 and episode != 0:
          ckpt_path = self.saver.save(self.sess, self.ckpts_path, global_step = self.step)
          print(color("\nStep: " + str(self.step) + ", Saver saved: " + ckpt_path, fg='white', bg='blue', style='bold'))

        if done:   
          break

      episode_len = s
      # add summary for all episodes
      total_mem_len += len(self.memory)
      avg_reward.append(episode_reward)
      self.learn()
      self.memory.clear()

      if episode % self.args.batch_size == 0 and episode != 0:
        avg_rew = np.mean(avg_reward)
        avg_reward.clear()
        avg_memory_len = float(total_mem_len) / float(self.args.batch_size)
        total_mem_len = 0
        summary = tf.Summary(value=[tf.Summary.Value(tag="avg reward", simple_value=avg_rew), tf.Summary.Value(tag="avg mem length", simple_value=avg_memory_len)])
        self.writer.add_summary(summary, global_step=episode)
        self.writer.flush()

        print(color("\n[Train] Avg Reward (30 rounds): " + "{:.2f}".format(avg_rew) + ", Avg Memory Length: " + "{:.2f}".format(avg_memory_len), fg='red', bg='white'))

      pbar.set_description("step: " + str(self.step) +  ", reward, " +  str(episode_reward) + ", episode length: " + str(episode_len))
      



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
    if self.step < self.args.observe_steps: # OBSERVE STAGE
      if random.random() > 0.5:
        return actions_in['up']
      else:
        return actions_in['down']

    prob = self.sess.run(self.up_prob, feed_dict={self.s: observation.reshape((1, -1))})

    if prob[0] > 0.5: # TRAIN STAGE
      return actions_in['up']
    else:
      return actions_in['down']

    # return self.env.get_random_action()     

