import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from AoI_env import *
from helper import *
import tensorflow as tf
import tflearn
import time

def run(max_user, user_num, t_factor, data_size, noise_sigma, mode, policy, round_l, round_h, max_episode, episode_len):

    for k in range(round_l,round_h):

        print('---------user_num:' + str(user_num) + '------------')
        print('---------k:' + str(k) + '------------')
        MAX_EPISODE = max_episode
        MAX_EPISODE_LEN = episode_len
      
        NUM_T = 1
        NUM_R = 4
        SIGMA2 = 1e-9
        # -------------------------------------------------
        # path for varying user
        res_path = 'Data_t_'+str(t_factor)+'_σ_'+str(noise_sigma)+'/'+ mode + '/' + policy +'/varying_usernum/usernum_' + str(user_num) + '/'
        #--------------------------------------------------
        #path for varying datasize
        # res_path = 'Data_t_'+str(t_factor)+'_σ_'+str(noise_sigma)+'/'+ mode + '/' + policy +'/varying_datasize/datasize_' + str(data_size) + '/'
        model_fold = 'Model_t_'+str(t_factor)+'_σ_'+str(noise_sigma)+'/'+'/varying_usernum/usernum_' + str(user_num) + '/'
        model_path = model_fold + str(k) 

        if not os.path.exists(res_path):
            os.mkdir(res_path) 
        if not os.path.exists(model_fold):
            os.mkdir(model_fold) 

        init_path = ''
        init_seqCnt = 40

        np.random.seed(1)
        
        #固定6个位置 对比的时候 user数目加一 是从原来数目user在原来的位置上加上后来的位置
        users_dis = np.random.uniform(50, 100, [1,max_user])

        user_common_config = {'state_dim':3, 'action_dim':1,
                                'model':'AR', 'num_r':NUM_R,  
                                'action_bound':2, 'data_buf_size':100, 
                                'sigma2':SIGMA2, 'data_size':data_size,
                                'dis':users_dis}
                                    
        train_config = {'minibatch_size':64,            'actor_lr':0.0001,     
                        'tau':0.001,                    'critic_lr':0.001,    
                        'gamma':0.99,                   'buffer_size':250000, 
                        'noise_sigma':noise_sigma,      'init_path' : init_path, 
                        't_factor':t_factor,            'user_num':user_num,
                        'random_seed':int(time.perf_counter()*1000%1000),
                        'episode_len':MAX_EPISODE_LEN}

        tf.compat.v1.reset_default_graph()

        # 0. initialize the session object
        sess = tf.compat.v1.Session() 

        # 1. include all user in the system according to the user_config
        user_config = []
        user_list = []

        for i in range(0,user_num):
            user_id = {'id':str(i)}
            user_config.append(user_id)
        for info in user_config:
                info.update(user_common_config)
                info['model_path'] = model_path
                info['meta_path'] = info['model_path']+'.meta'
                info['init_path'] = init_path
                info['init_seqCnt'] = init_seqCnt
                user_list.append(MecTerm(info, train_config))
        # print('------------ Initialization OK! ------------')
        
        # 2. create the simulation env
        if policy=='ddpg' and mode=='train':
        	env = MecSvrEnv(MecTermRL(sess, user_list, user_common_config, train_config))
        	sess.run(tf.compat.v1.global_variables_initializer())
	        tflearn.config.is_training(is_training=False, session=sess)
	        env.init_target_network()

        elif policy=='ddpg' and mode=='test':
        	env = MecSvrEnv(MecTermRL_test(sess, user_list, user_common_config, train_config))

        elif policy=='random':
        	env = MecSvrEnv(Random_policy(user_list, user_common_config, train_config))
                
        elif policy=='GA':
            env = MecSvrEnv(GA_policy(user_list, user_common_config, train_config))

        res_r = []
        res_p = []
        res_e = []
        res_AoI = []

        # 3. start to explore for each episode
        for i in range(MAX_EPISODE):
            #reset env
            env.reset()

            #record parameter values of each episode
            cur_r_ep = 0
            cur_p_ep = 0
            cur_e_ep = 0
            cur_AoI_ep = 0

            for j in range(MAX_EPISODE_LEN):
                _, _, cur_r, cur_p, cur_e, cur_AoI, done = env.step_transmit()
                
                cur_r_ep += cur_r
                cur_p_ep += cur_p
                cur_e_ep += cur_e
                cur_AoI_ep += cur_AoI

                if done:
                    res_r.append(cur_r_ep/MAX_EPISODE_LEN)
                    res_p.append(cur_p_ep/MAX_EPISODE_LEN)
                    res_e.append(cur_e_ep/MAX_EPISODE_LEN)
                    res_AoI.append(cur_AoI_ep/MAX_EPISODE_LEN)

                    # print('%d:reward:%s,power:%s,energy:%s,AoI:%s'%(i,cur_r_ep/MAX_EPISODE_LEN,cur_p_ep/MAX_EPISODE_LEN,cur_e_ep/MAX_EPISODE_LEN,cur_AoI_ep/MAX_EPISODE_LEN))

        name = res_path + 't_'+ str(t_factor) + 'σ_' + str(noise_sigma) + '_' + str(k) + '_' + time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime(time.time()))
        np.savez(name, res_r, res_p, res_e, res_AoI)

        if mode=='train':   
	        tflearn.config.is_training(is_training=False, session=sess)

	        #Create a saver object which will save all the variables
	        saver = tf.train.Saver() 
	        saver.save(sess, model_path)
       
        sess.close()


# save the data of sample intervals and queue delays in test stage
def run_interval(max_user, user_num, t_factor, data_size, noise_sigma, mode, policy, round_l, round_h, max_episode, episode_len):

    for k in range(round_l,round_h):

        print('---------user_num:' + str(user_num) + '------------')
        print('---------k:' + str(k) + '------------')
        MAX_EPISODE = max_episode
        MAX_EPISODE_LEN = episode_len
      
        NUM_T = 1
        NUM_R = 4
        SIGMA2 = 1e-9

        res_path = 'Data_t_'+str(t_factor)+'_σ_'+str(noise_sigma)+'/'+ mode + '/' + policy +'/varying_usernum/usernum_' + str(user_num) + '/'
        interval_path = res_path + 'interval/'

        model_fold = 'Model_t_'+str(t_factor)+'_σ_'+str(noise_sigma)+'/'+'/varying_usernum/usernum_' + str(user_num) + '/'
        model_path = model_fold + str(k) 

        if not os.path.exists(res_path):
            os.mkdir(res_path) 
        if not os.path.exists(interval_path):
            os.mkdir(interval_path) 
        if not os.path.exists(model_fold):
            os.mkdir(model_fold) 

        init_path = ''
        init_seqCnt = 40

        np.random.seed(1)
        
        #固定6个位置 对比的时候 user数目加一 是从原来数目user在原来的位置上加上后来的位置
        users_dis = np.random.uniform(50, 100, [1,max_user])

        user_common_config = {'state_dim':3, 'action_dim':1,
                                'model':'AR', 'num_r':NUM_R,  
                                'action_bound':2, 'data_buf_size':100, 
                                'sigma2':SIGMA2, 'data_size':data_size,
                                'dis':users_dis}
                                    
        train_config = {'minibatch_size':64,            'actor_lr':0.0001,     
                        'tau':0.001,                    'critic_lr':0.001,    
                        'gamma':0.99,                   'buffer_size':250000, 
                        'noise_sigma':noise_sigma,      'init_path' : init_path, 
                        't_factor':t_factor,            'user_num':user_num,
                        'random_seed':int(time.perf_counter()*1000%1000),
                        'episode_len':MAX_EPISODE_LEN}

        tf.compat.v1.reset_default_graph()

        # 0. initialize the session object
        sess = tf.compat.v1.Session() 

        # 1. include all user in the system according to the user_config
        user_config = []
        user_list = []

        for i in range(0,user_num):
            user_id = {'id':str(i)}
            user_config.append(user_id)
        for info in user_config:
                info.update(user_common_config)
                info['model_path'] = model_path
                info['meta_path'] = info['model_path']+'.meta'
                info['init_path'] = init_path
                info['init_seqCnt'] = init_seqCnt
                user_list.append(MecTerm(info, train_config))
        # print('------------ Initialization OK! ------------')
        
        # 2. create the simulation env
        if policy=='ddpg' and mode=='train':
            env = MecSvrEnv(MecTermRL(sess, user_list, user_common_config, train_config))
            sess.run(tf.compat.v1.global_variables_initializer())
            tflearn.config.is_training(is_training=False, session=sess)
            env.init_target_network()

        elif policy=='ddpg' and mode=='test':
            env = MecSvrEnv(MecTermRL_test(sess, user_list, user_common_config, train_config))

        elif policy=='random':
            env = MecSvrEnv(Random_policy(user_list, user_common_config, train_config))

        elif policy=='max':
            env = MecSvrEnv(Maxpower_policy(user_list, user_common_config, train_config))

        res_r = []
        res_p = []
        res_e = []
        res_AoI = []

        #sample or update in each slot or not
        res_smp_lst = []
        res_upd_lst = []

        #sample interval and queue delay list of each episode
        smp_intvl_lst  = []
        queue_delay_lst = []

        # 3. start to explore for each episode
        for i in range(MAX_EPISODE):
            #reset env
            env.reset()

            #record parameter values of each episode
            cur_r_ep = 0
            cur_p_ep = 0
            cur_e_ep = 0
            cur_AoI_ep = 0

            for j in range(MAX_EPISODE_LEN):
                smp_lst, upd_lst, _, _, _, _, done = env.step_transmit()
                res_smp_lst.append(smp_lst)
                res_upd_lst.append(upd_lst)

                if done:
                    smp_intvl, queue_delay =  compute_smp_interval_N_queue_delay(res_smp_lst, res_upd_lst, user_num)
                    smp_intvl_lst.append(smp_intvl)
                    queue_delay_lst.append(queue_delay)

        #save sample interval and queue delay
        name_inter = interval_path + str(k)
        np.savez(name_inter, smp_intvl_lst, queue_delay_lst)

        if mode=='train':   
            tflearn.config.is_training(is_training=False, session=sess)

            #Create a saver object which will save all the variables
            saver = tf.train.Saver() 
            saver.save(sess, model_path)
        sess.close()


def compute_smp_interval_N_queue_delay(smp_lst,upd_lst,user_num):
    smp_intvl = 50/np.sum(smp_lst)/user_num
    cur_queue_delay = np.zeros(user_num)
    queue_delay_tuple = []

    for i in range(0, len(smp_lst)):
        for y in range(user_num):
            if smp_lst[i][y] == 1:
                cur_queue_delay[y] = 0
            if upd_lst[i][y] == 1:
                queue_delay_tuple.append(cur_queue_delay[y])
                cur_queue_delay[y] = 0
            if upd_lst[i][y] == 0:
                cur_queue_delay[y] += 0.1
    queue_delay = np.sum(queue_delay_tuple)/user_num
    return smp_intvl, queue_delay 