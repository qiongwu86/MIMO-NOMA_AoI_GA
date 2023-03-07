from running_term import *
from multiprocessing import Process

#parameters
train = 'train'
test = 'test'
ddpg = 'ddpg'
random = 'random'
maxpower = 'max'
GA_policy = 'GA'

# for varying user numbers
# user_num_list = [2,3,4,5,6]
user_num_list = [6]

#for varying datasize
# datasize_list = [1000,2000,3000,4000,5000]

data_size = 1000
t_factor = 0.5
noise_sigma = 0.004

max_episode = 800
episode_len = 500
run_round_l = 0
run_round_h = 10 

if __name__ == '__main__':
    proc_train_1 = []
    proc_train_2 = []
    proc_train_3 = []
    proc_train_4 = []
    proc_train_5 = []

    proc_RL_test_1 = []
    proc_RL_test_2 = []
    proc_RL_test_3 = []

    proc_random = []
    proc_max = []
    proc_RL_test = []

    # for user_num in user_num_list:
    #     #固定6个位置 对比的时候 user数目加一 是从原来数目user在原来的位置上加上后来的位置
    #     max_user = 6
    #     proc_train_1.append(Process(target = run, args=(max_user, user_num, t_factor, data_size, noise_sigma, train, ddpg, 0, 2, max_episode, episode_len,)))
    #     proc_train_2.append(Process(target = run, args=(max_user, user_num, t_factor, data_size, noise_sigma, train, ddpg, 2, 4, max_episode, episode_len,)))
    #     proc_train_3.append(Process(target = run, args=(max_user, user_num, t_factor, data_size, noise_sigma, train, ddpg, 4, 6, max_episode, episode_len,)))
    #     proc_train_4.append(Process(target = run, args=(max_user, user_num, t_factor, data_size, noise_sigma, train, ddpg, 6, 8, max_episode, episode_len,)))
    #     proc_train_5.append(Process(target = run, args=(max_user, user_num, t_factor, data_size, noise_sigma, train, ddpg, 8, 10, max_episode, episode_len,)))


    #     # proc_random.append(Process(target = run_interval, args=(max_user, user_num, t_factor, data_size, noise_sigma, test, random, run_round_l, run_round_h, max_episode, episode_len,)))
    #     proc_RL_test_1.append(Process(target = run, args=(max_user, user_num, t_factor, data_size, noise_sigma, test, ddpg, 0, 4, max_episode, episode_len,)))
    #     proc_RL_test_2.append(Process(target = run, args=(max_user, user_num, t_factor, data_size, noise_sigma, test, ddpg, 4, 7, max_episode, episode_len,)))
    #     proc_RL_test_3.append(Process(target = run, args=(max_user, user_num, t_factor, data_size, noise_sigma, test, ddpg, 7, 10, max_episode, episode_len,)))

    # for p_train_1,p_train_2,p_train_3,p_train_4,p_train_5 in zip(proc_train_1, proc_train_2, proc_train_3, proc_train_4, proc_train_5):
    #     p_train_1.start()
    #     p_train_2.start()
    #     p_train_3.start()
    #     p_train_4.start()
    #     p_train_5.start()

    # for p_train_1,p_train_2,p_train_3 in zip(proc_train_1, proc_train_2, proc_train_3):
    #     p_train_1.join()
    #     p_train_2.join()
    #     p_train_3.join()

    # for p_RL_test_1,p_RL_test_2,p_RL_test_3 in zip(proc_RL_test_1, proc_RL_test_2):
    #     p_RL_test_1.start()
    #     p_RL_test_2.start()
    #     p_RL_test_3.start()

# ----------------------------------------------------------------------------------------------------------------
#for single user_num training or test
    # for x in range(0,10):
    #     proc_train_1.append(Process(target = run, args=(6, 2, t_factor, data_size, noise_sigma, train, ddpg, x, x+1, max_episode, episode_len,)))

    # for p_train_1 in proc_train_1:
    #     p_train_1.start()

#--------------------------------------------------------------------------------------------------------------------
#for varying data size test with user num 2
data_size_lst = [1000, 2000, 3000, 4000, 5000]
for data_size in data_size_lst:
    proc_RL_test_1 = []
    for x in range(0,10):
        proc_RL_test_1.append(Process(target = run, args=(6, 3, t_factor, data_size, noise_sigma, test, GA_policy, x, x+1, max_episode, episode_len,)))
    # proc_RL_test_1.append(Process(target = run, args=(6, 2, t_factor, data_size, noise_sigma, test, ddpg, 0, 4, max_episode, episode_len,)))
    # proc_RL_test_2.append(Process(target = run, args=(6, 2, t_factor, data_size, noise_sigma, test, ddpg, 4, 7, max_episode, episode_len,)))
    # proc_RL_test_3.append(Process(target = run, args=(6, 2, t_factor, data_size, noise_sigma, test, ddpg, 7, 10, max_episode, episode_len,)))

    # for p1 in proc_RL_test_1:
    #     p1.start()

#------------------------------------join() in case next datasize process arise error 
    # for p1 in proc_RL_test_1:
    #     p1.join()     