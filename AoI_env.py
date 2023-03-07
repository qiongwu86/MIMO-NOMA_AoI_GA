import numpy as np
# from helper import *
import tensorflow as tf
from GA import *

class MecTerm(object):
    """
    MEC terminal parent class
    """
    def __init__(self, user_config, train_config):
        self.train_config = train_config
        self.user_config = user_config
        self.data_size = user_config['data_size'] #unit:bit
        self.id = user_config['id']
        self.dis = user_config['dis'][0][int(self.id)]
        self.action_dim = user_config['action_dim']
        self.t_factor = train_config['t_factor']
        self.sigma2 = user_config['sigma2']
        self.num_user = train_config['user_num']
        self.action_bound = user_config['action_bound']

        self.init_path = ''
        self.init_seqCnt = 0

        self.n_t = 1
        self.n_r = user_config['num_r']    
        self.DataBuf = 0
        
        self.SINR = 0
        self.Power = np.zeros(self.action_dim)
        self.State_user = []

        # some pre-defined parameters
        self.C_s = 0.5
        self.t = 0.1 #unit:second
        self.bandwidth = 18 #kHz
        self.channelModel = ARModel(self.n_t, self.n_r, seed=train_config['random_seed'])

        # parameters of AoI 
        self.phi = 0
        self.Phi_BS = 0
        self.upd = 0
        self.sample = 0

    def sampleCh(self):
        self.Channel = self.channelModel.sampleCh(self.dis)
        return self.Channel

    def getCh(self):
        self.Channel = self.channelModel.getCh(self.dis)    
        return self.Channel
   
    def setSINR(self, sinr):
        self.SINR = sinr
        self.sampleCh()
        channel_gain = np.power(np.linalg.norm(self.Channel),2)/self.sigma2
        
    def AoI_upd(self, powers, sinr):
        self.SINR = sinr
        self.Power = powers[0]
        self.trans_rate = self.bandwidth*np.log2(1 + self.SINR)*1000 #unit:bit/s

        if self.trans_rate*self.t > self.DataBuf:
        # self.Power >= self.action_bound/(self.action_level-1):
            self.upd = 1
        else:
            self.upd = 0

        self.compute_samlple()
        if self.sample == 1:
            self.phi = 0
            self.DataBuf = self.data_size
        elif self.sample == 0:
            self.phi += self.t

        if self.upd == 1:
            self.trans_time = self.DataBuf/self.trans_rate
            self.Phi_BS = self.phi+self.trans_time
            self.DataBuf = 0
        else:
            self.Phi_BS += self.t
            self.trans_time = 0


        def upd_fitness(self, sinr):
            # For GA fitness computing without user update 
            DataBuf = self.DataBuf
            trans_rate = self.bandwidth*np.log2(1 + sinr)*1000

            if trans_rate*self.t > DataBuf:
                upd = 1
            else:
                upd = 0

            if self.t_factor*self.C_s - (1-self.t_factor)*upd*(self.phi+self.t) < 0:
                sample = 1
                phi_ = 0
                DataBuf = self.data_size
            else:
                sample = 0
                phi_ = self.phi + self.t

            if upd == 1:
                trans_time = DataBuf/trans_rate
                Phi_BS_ = phi_ + trans_time
                DataBuf = 0
            else:
                Phi_BS_ = self.Phi_BS+self.t
                trans_time = 0
            return trans_time, sample,  Phi_BS_


    def print_info(self):
        print ('id:%d\ttrans_rate:%s\t\ttrans_time:%s\t\tbuff:%s\t\tpower:%s\t\tupd:%d\tsample:%d\tphi:%s\t\tPhi_BS:%s'
            %(int(self.id),self.trans_rate,self.trans_time,self.DataBuf,self.Power,self.upd,self.sample,self.phi,self.Phi_BS))

    def buffer_reset(self, seqCount):
        self.DataBuf = 0
        self.sampleCh()
        if seqCount >= self.init_seqCnt:
            self.isUpdateActor = True

        self.phi = 0
        self.Phi_BS = 0
        self.upd = 0
        self.sample = 0
        return self.DataBuf

    def compute_samlple(self):
        # print ('id:%d\ttau*C_s:%s\t(1-tau)*upd*(phi+t_slot):%s'
        #     %(int(self.id), self.t_factor*self.C_s, (1-self.t_factor)*self.upd*(self.phi+self.t)))
        if self.t_factor*self.C_s - (1-self.t_factor)*self.upd*(self.phi+self.t) < 0:
            self.sample = 1
        else:
            self.sample = 0

class Random_policy(object):
    """docstring for random"""
    def __init__(self, user_list, user_config, train_config): 
        self.user_list = user_list
        self.user_config = user_config
        self.train_config = train_config
        self.action_bound = user_config['action_bound']
        self.t_factor = train_config['t_factor']
        self.num_user = train_config['user_num']
        self.action_dim = user_config['action_dim']
        self.powers = []
        self.State = self.global_state()

    def global_state(self):
        State = []
        for user in self.user_list:
            State = np.append(State, np.array([user.upd, user.SINR, user.Phi_BS]))
        return State

    def global_action(self):
        Action = []
        for user in self.user_list:
            Action = np.append(Action, user.Power)
        return Action

    def global_smp_upd(self):
        smp_lst = []
        upd_lst = []
        for user in self.user_list:
            smp_lst = np.append(smp_lst, user.sample)
            upd_lst = np.append(upd_lst, user.upd)
        return smp_lst, upd_lst

    def feedback(self, sinr_list, powers, done):
        self.powers = powers
        self.sinr_list = sinr_list
        self.next_state = []
        self.power_CnSpt = 0
        self.energy_CnSpt = 0
        self.Reward = 0
        self.total_AOI = 0

        #in order to save the data of sample intervel and queue delay
        smp_lst, upd_lst = self.global_smp_upd()

        for user in self.user_list:
            #update AoI of each user
            user.AoI_upd(self.powers[int(user.id)], self.sinr_list[int(user.id)])

            # get the reward for the current slot
            self.Reward += - (self.t_factor*(user.Power*user.trans_time+user.sample*user.C_s) + (1-self.t_factor)*user.Phi_BS)
            # print('Phi_BS %s:%s'% (user.id,user.Phi_BS))
            self.power_CnSpt += user.Power
            self.energy_CnSpt += user.Power*user.trans_time+user.sample*user.C_s
            self.total_AOI += user.Phi_BS

            # estimate the channel for next slot
            user.sampleCh()

            # Update State
            next_state = self.global_state()

            # Collect the action of all users
            self.Action = self.global_action()

        self.State = next_state
        return  self.State, self.Action, smp_lst, upd_lst, self.Reward, self.power_CnSpt, self.energy_CnSpt, self.total_AOI

    def predict(self, isRandom):
        power = self.action_bound*np.random.random(size=(1,self.num_user*self.user_config['action_dim']))[0]
        noise = np.zeros(self.num_user*self.user_config['action_dim'])[0]
        return power, noise

    
class GA_policy(Random_policy):
    def evaluate(self, sinr_list, powers):
        fitness = 0
        for i, user in enumerate(self.user_list):
            trans_time, sample,  Phi_BS_ = user.upd_fitness(sinr_list)
            fitness +=  - (self.t_factor*(powers[i]*trans_time + sample*user.C_s) + (1-self.t_factor)*Phi_BS_)
        return fitness



class MecTermRL_test(object):
    """
    MEC terminal class using RL
    """
    def __init__(self, sess, user_list, user_config, train_config): 
        self.sess = sess
        self.user_list = user_list
        self.user_config = user_config
        self.train_config = train_config
        self.action_bound = user_config['action_bound']
        self.t_factor = train_config['t_factor']
        self.num_user = train_config['user_num']
        self.isUpdateActor = True
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.powers = []
        self.State = self.global_state()

        saver = tf.train.import_meta_graph(self.user_list[0].user_config['meta_path'])
        saver.restore(sess, self.user_list[0].user_config['model_path'])
 
        graph = tf.get_default_graph()
        input_str = "input_" + "/X:0"
        output_str = "output_" + ":0"
        self.inputs = graph.get_tensor_by_name(input_str)
        if not 'action_level' in user_config:
            self.out = graph.get_tensor_by_name(output_str)

    def global_state(self):
        State = []
        for user in self.user_list:
            State = np.append(State, np.array([user.upd, user.SINR, user.Phi_BS]))
        return State

    def global_action(self):
        Action = []
        for user in self.user_list:
            Action = np.append(Action, user.Power)
        return Action

    def global_smp_upd(self):
        smp_lst = []
        upd_lst = []
        for user in self.user_list:
            smp_lst = np.append(smp_lst, user.sample)
            upd_lst = np.append(upd_lst, user.upd)
        return smp_lst, upd_lst

    def feedback(self, sinr_list, powers, done):
        self.powers = powers
        self.sinr_list = sinr_list
        self.next_state = []
        self.power_CnSpt = 0
        self.energy_CnSpt = 0
        self.Reward = 0
        self.total_AOI = 0
        #in order to save the data of sample intervel and queue delay
        smp_lst, upd_lst = self.global_smp_upd()

        for user in self.user_list:
            #update AoI of each user
            user.AoI_upd(self.powers[int(user.id)], self.sinr_list[int(user.id)])

            # get the reward for the current slot
            self.Reward += - (self.t_factor*(user.Power*user.trans_time+user.sample*user.C_s) + (1-self.t_factor)*user.Phi_BS)
            # print('Phi_BS %s:%s'% (user.id,user.Phi_BS))
            self.power_CnSpt += user.Power
            self.energy_CnSpt += user.Power*user.trans_time+user.sample*user.C_s
            self.total_AOI += user.Phi_BS

            # estimate the channel for next slot
            user.sampleCh()

            # Update State
            next_state = self.global_state()

            # Collect the action of all users
            self.Action = self.global_action()

        self.State = next_state
        return  self.State, self.Action, smp_lst, upd_lst, self.Reward, self.power_CnSpt, self.energy_CnSpt, self.total_AOI

    def predict(self, isRandom):
        power = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim*self.num_user))})[0]
        return power, np.zeros(self.action_dim*self.num_user)

class MecTermRL(object):
    """
    MEC terminal class using RL
    """
    def __init__(self, sess, user_list, user_config, train_config): 
        self.sess = sess
        self.user_list = user_list
        self.user_config = user_config
        self.train_config = train_config
        self.agent = DDPGAgent(sess, user_config, train_config)
        self.action_bound = user_config['action_bound']
        self.t_factor = train_config['t_factor']
        self.num_user = train_config['user_num']
        self.isUpdateActor = True
        self.action_dim = user_config['action_dim']
        self.powers = []
        self.State = self.global_state()

    def global_state(self):
        State = []
        for user in self.user_list:
            State = np.append(State, np.array([user.upd, user.SINR, user.Phi_BS]))
        return State

    def global_action(self):
        Action = []
        for user in self.user_list:
            Action = np.append(Action, user.Power)
        return Action

    def global_smp_upd(self):
        smp_lst = []
        upd_lst = []
        for user in self.user_list:
            smp_lst = np.append(smp_lst, user.sample)
            upd_lst = np.append(upd_lst, user.upd)
        return smp_lst, upd_lst


    def feedback(self, sinr_list, powers, done):
        self.powers = powers
        self.sinr_list = sinr_list
        self.next_state = []
        self.power_CnSpt = 0
        self.energy_CnSpt = 0
        self.Reward = 0
        self.total_AOI = 0

        smp_lst, upd_lst = self.global_smp_upd()

        for user in self.user_list:
            #update AoI of each user
            user.AoI_upd(self.powers[int(user.id)], self.sinr_list[int(user.id)])

            # get the reward for the current slot
            self.Reward += - (self.t_factor*(user.Power*user.trans_time+user.sample*user.C_s) + (1-self.t_factor)*user.Phi_BS)
            # print('Phi_BS %s:%s'% (user.id,user.Phi_BS))
            self.power_CnSpt += user.Power
            self.energy_CnSpt += user.Power*user.trans_time+user.sample*user.C_s
            self.total_AOI += user.Phi_BS

            # estimate the channel for next slot
            user.sampleCh()

            # Update State
            next_state = self.global_state()

            # Collect the action of all users
            self.Action = self.global_action()

        self.agent.update(self.State, self.Action, self.Reward, done, next_state, self.isUpdateActor)

        self.State = next_state
        return  self.State, self.Action, smp_lst, upd_lst, self.Reward, self.power_CnSpt, self.energy_CnSpt, self.total_AOI

    def predict(self, isRandom):
        power, noise = self.agent.predict(self.State, self.isUpdateActor)
        power = np.fmax(0, np.fmin(self.action_bound, power))
        return power, noise
    
class MecSvrEnv(object):
    """
    Simulation environment
    """
    def __init__(self, Agent_Term): 
        self.Agent_Term = Agent_Term
        self.sigma2 = self.Agent_Term.user_config['sigma2']
        self.max_len = self.Agent_Term.train_config['episode_len']
        self.count = 0
        self.seqCount = 0
        
    def init_target_network(self):
        self.Agent_Term.agent.init_target_network()


    def step_transmit(self, isRandom=True):
        # get the channel vectors 
        channels = np.transpose([user.getCh() for user in self.Agent_Term.user_list])

        # print(self.Agent_Term isinstance GA_policy)
        if isinstance(self.Agent_Term, GA_policy):
            CXPB, MUTPB, NGEN, popsize = 0.8, 0.5, 100, 100
            up = [2]*self.Agent_Term.num_user
            low = [0]*self.Agent_Term.num_user
            parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
            run = GA(parameter)
            run.GA_main()
        
        def evaluate(geneinfo):
            powers = np.array(geneinfo).reshape(self.Agent_Term.num_user, self.Agent_Term.action_dim)
            sinr_list = self.compute_sinr(channels, powers[:,0])
            fitness = self.Agent_Term.evaluate(sinr_list, powers)
            return fitness

        # get the transmit powers 
        powers, _ = self.Agent_Term.predict(isRandom)
        powers = np.reshape(powers,(self.Agent_Term.num_user, self.Agent_Term.action_dim))

        # compute the sinr for each user
        sinr_list = self.compute_sinr(channels, powers[:,0])
        # print (sinr_list)

        # feedback the sinr to each user
        _, _, smp_lst, upd_lst, reward, power_CnSpt, energy_CnSpt, total_AOI = self.Agent_Term.feedback(sinr_list, powers, self.count >= self.max_len)

        self.count += 1

        #print information for debug
        # self.users_info()

        return smp_lst, upd_lst, reward, power_CnSpt, energy_CnSpt, total_AOI ,self.count >= self.max_len

    def users_info(self):
        if self.count >= self.max_len:
            for user in self.Agent_Term.user_list:
                user.print_info() 

    def compute_sinr(self, channels, powers):
        # Power-Domain NOMA 
        # calculate the received power at the BS for each user
        channel_gains = np.power(np.linalg.norm(channels, axis=0), 2)

        receive_powers = []
        for i in range(self.Agent_Term.num_user):
            receive_power = channel_gains[i]*powers[i]
            receive_powers.append(receive_power)
        total_power = np.sum(receive_powers)

        # ordering the channels by their power gain in an acending order
        idx_list = np.argsort(receive_powers)[::-1]

        # get access to the channel and decode in an decending order
        sinr_list = np.zeros(self.Agent_Term.num_user)
        for i in range(self.Agent_Term.num_user):
            user_idx = idx_list[i]
            total_power -= receive_powers[user_idx]
            sinr_list[user_idx] = receive_powers[user_idx]/(total_power+self.sigma2)
        return sinr_list

    def reset(self, isTrain=True):
        self.count = 0
        if isTrain:
            init_data_buf_size = [user.buffer_reset(self.seqCount) for user in self.Agent_Term.user_list]
            # get the channel vectors   
            channels = np.transpose([user.getCh() for user in self.Agent_Term.user_list])
            # compute the sinr for each user
            powers=np.ones((1,self.Agent_Term.num_user))[0]
            sinr_list = self.compute_sinr(channels,powers)
        else:
            init_data_buf_size = 0 
            sinr_list = [0 for user in self.Agent_Term.user_list]

        for i in range(self.Agent_Term.num_user):
            self.Agent_Term.user_list[i].setSINR(sinr_list[i])

        self.seqCount += 1
        return init_data_buf_size
    