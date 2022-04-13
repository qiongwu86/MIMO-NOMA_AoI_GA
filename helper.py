import os
import numpy as np
import tensorflow as tf
from ddpg_lib import * 
import ipdb as pdb
import matplotlib.pyplot as plt

alpha = 2.0
ref_loss = 0.001
width_lane = 5
Hight_RSU = 10 

def complexGaussian(row=1, col=1, amp=1.0):
    real = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    img = np.random.normal(size=[row,col])[0]*np.sqrt(0.5)
    return amp*(real + 1j*img)

class ARModel(object):
    """docstring for AR channel Model"""
    def __init__(self, n_t=1, n_r=1, rho=0.95, seed=123):
        self.n_t = n_t
        self.n_r = n_r
        np.random.seed([seed])       
        self.rho = rho
        self.H = complexGaussian(self.n_t, self.n_r)  
        
    def getCh(self,dis):
        self.dis = dis
        self.path_loss = ref_loss/np.power(np.linalg.norm(self.dis),2)
        return self.H*np.sqrt(self.path_loss)
        
    def sampleCh(self,dis):
        self.H = self.rho*self.H + complexGaussian(self.n_t, self.n_r, np.sqrt(1-self.rho*self.rho))
        return self.getCh(dis)

class DQNAgent(object):
    """docstring for DQNAgent"""
    def __init__(self, sess, user_config, train_config):
        self.sess = sess
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.num_user = train_config['user_num']
        self.action_bound = user_config['action_bound']
        self.action_level = user_config['action_level']
        self.minibatch_size = int(train_config['minibatch_size'])
        self.epsilon = float(train_config['epsilon'])
        
        self.state_nums = self.state_dim*self.num_user
        self.action_nums = 1
        for i in range(self.num_user):
            self.action_nums *= (self.action_level)
        
        self.max_step = 500000
        self.pre_train_steps = 25000
        self.total_step = 0
        self.DQN = DeepQNetwork(sess, self.state_nums, self.action_nums, float(train_config['critic_lr']), float(train_config['tau']), float(train_config['gamma']))
        self.replay_buffer = ReplayBuffer(int(train_config['buffer_size']), int(train_config['random_seed']))

    def init_target_network(self):
        self.DQN.update_target_network()
        
    def predict(self, s):
        if self.total_step <= self.max_step:
            self.epsilon *= 0.9999999999999999999976
        # print (self.epsilon)
        # print (np.random.rand(1) < self.epsilon or self.total_step < self.pre_train_steps)
        if np.random.rand(1) < self.epsilon or self.total_step < self.pre_train_steps:
            action = np.random.randint(0, self.action_nums)
        else:
            action, _ = self.DQN.predict(np.reshape(s, (1, self.state_nums)))
        self.total_step += 1
        # print ('self.total_step:',self.total_step)
        # print (' self.epsilon:', self.epsilon)
        return action, np.zeros([1])
    
    def update(self, s, a, r, t, s2):
        self.replay_buffer.add(np.reshape(s, (self.state_nums,)), a, r,
                              t, np.reshape(s2, (self.state_nums,)))
        
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    self.replay_buffer.sample_batch(self.minibatch_size)

            # calculate targets
            _, q_out = self.DQN.predict(s_batch)
            target_prediction, target_q_out = self.DQN.predict_target(s2_batch)
            
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    q_out[k][a_batch[k]] = r_batch[k]
                else:
                    q_out[k][a_batch[k]] = r_batch[k] + self.DQN.gamma * target_q_out[k][target_prediction[k]]

            # Update the critic given the targets
            q_loss, _ = self.DQN.train(
                s_batch, q_out) 
            
            # losses.append(q_loss)
            # Update target networks
            self.DQN.update_target_network()

class DDPGAgent(object):
    """docstring for DDPGAgent"""
    def __init__(self, sess, user_config, train_config):
        self.sess = sess
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.num_user = train_config['user_num']
        self.action_bound = user_config['action_bound']
        self.init_path = train_config['init_path'] if 'init_path' in train_config else ''

        # self.state_nums = self.state_dim*self.num_user
        # self.action_nums = self.action_dim*self.num_user

        self.minibatch_size = int(train_config['minibatch_size'])
        self.noise_sigma = float(train_config['noise_sigma'])

        # initalize the required modules: actor, critic and replaybuffer
        self.actor = ActorNetwork(sess, self.state_dim*self.num_user, self.action_dim*self.num_user, self.action_bound, float(train_config['actor_lr']), float(train_config['tau']), self.minibatch_size)
        
        self.critic = CriticNetwork(sess, self.state_dim*self.num_user, self.action_dim*self.num_user, float(train_config['critic_lr']), float(train_config['tau']), float(train_config['gamma']), self.actor.get_num_trainable_vars())
        
        self.replay_buffer = ReplayBuffer(int(train_config['buffer_size']), int(train_config['random_seed']))

        # mu, sigma=0.12, theta=.15, dt=1e-2,
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim*self.num_user), sigma=self.noise_sigma) 
        # self.actor_noise = GaussianNoise(0.1, 0.01, size=np.array([self.action_dim]))
        
    def init_target_network(self):
        # Initialize the original network and target network with pre-trained model
        if len(self.init_path) == 0:
            self.actor.update_target_network()
        else:
            self.actor.init_target_network(self.init_path)
        self.critic.update_target_network()

    # input current state and then return the next action
    def predict(self, s, isUpdateActor):
        if isUpdateActor:
            # self.sigma *= 0.99995
            # noise =  OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim*self.num_user),sigma=self.noise_sigma)()
            noise = self.actor_noise()
        else:
            noise = np.zeros(self.action_dim)
            
        return self.actor.predict(np.reshape(s, (1, self.actor.s_dim)))[0] + noise, noise
        # return self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) + np.random.normal(0.0,0.1,[self.action_dim])

    def update(self, s, a, r, t, s2, isUpdateActor):
        self.replay_buffer.add(np.reshape(s, (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), r,
                              t, np.reshape(s2, (self.actor.s_dim,)))
        
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    self.replay_buffer.sample_batch(self.minibatch_size)

            # calculate targets
            target_q = self.critic.predict_target(
                    s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(
                s_batch, a_batch, np.reshape(y_i, (self.minibatch_size, 1)))

            if isUpdateActor:
                # Update the actor policy using the sampled gradient
                a_outs = self.actor.predict(s_batch)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()
    