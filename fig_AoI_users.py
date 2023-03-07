import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
# import ipdb as pdb
import matplotlib.pyplot as plt
# from helper import *
from matplotlib.font_manager import FontProperties  


def output_avg(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    for name in fileList[:]:
    	path = dir_path + name
    	res = np.load(path)
    	temp_rs = np.array(res['arr_3'])
    	avg_rs.append(temp_rs)

# for debug-----------to check each test result
    # n =  9
    # path = dir_path + fileList[n]
    # print (fileList,fileList[n])
    # res = np.load(path)
    # temp_rs = np.array(res['arr_3'])
    # avg_rs.append(temp_rs)

    avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
    return avg_rs      

user_num_list = [2,3,4,5,6]
# user_num_list = [2]
AoI_ddpg = []
AoI_max = []
AoI_random = []

Data_file = 'Data_t_0.5_σ_0.004'

for user_num in user_num_list:
    res_path_ddpg = Data_file + '/test/ddpg/varying_usernum/usernum_' + str(user_num) + '/'
    res_path_maxpower = Data_file + '/test/max/varying_usernum/usernum_' + str(user_num) + '/'
    res_path_random_power = Data_file + '/test/random/varying_usernum/usernum_' + str(user_num) + '/'
    AoI_ddpg.append(np.mean(output_avg(res_path_ddpg), axis=0))
    # AoI_max.append(np.mean(output_avg(res_path_maxpower), axis=0))
    AoI_random.append(np.mean(output_avg(res_path_random_power), axis=0))

x = np.arange(len(user_num_list))
width = 0.25

print (AoI_ddpg, AoI_random)
plt.bar(x, AoI_ddpg,  width=width, label='ddpg')
plt.bar(x +  width, AoI_random, width=width, label='random', color='darkred', tick_label=user_num_list)
# plt.bar(x + 2 *width, AoI_max, width=width, label='max', color='salmon')

plt.xticks()
plt.ylabel('AoI')
plt.xlabel('$M$')

plt.grid(linestyle=':')

# font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=15) 

# plt.legend(prop = font)

plt.legend()
plt.show()

# 0.5373234790872096  0.5122950156634629