import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
# import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *
from matplotlib.font_manager import FontProperties  

def output_avg_energy(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    for name in fileList:
        path = dir_path + name
        res = np.load(path)
        temp_rs = np.array(res['arr_2'])
        avg_rs.append(temp_rs)
    avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
    return avg_rs

def output_avg_AoI(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    for name in fileList:
        path = dir_path + name
        res = np.load(path)
        temp_rs = np.array(res['arr_3'])
        avg_rs.append(temp_rs)
    avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
    return avg_rs

data_size_lst = [1000, 2000, 3000, 4000, 5000]

AoI_ddpg = []
energy_ddpg = []
AoI_random = []
energy_random = []

Data_file = 'Data_t_0.5_σ_0.004'

for data_size in data_size_lst:
    res_path_ddpg = Data_file + '/test/ddpg/varying_datasize/datasize_' + str(data_size) + '/' 
    res_path_random = Data_file + '/test/random/varying_datasize/datasize_' + str(data_size) + '/' 

    AoI_ddpg.append(np.mean(output_avg_AoI(res_path_ddpg),axis=0))
    energy_ddpg.append(np.mean(output_avg_energy(res_path_ddpg),axis=0))

    AoI_random.append(np.mean(output_avg_AoI(res_path_random),axis=0))
    energy_random.append(np.mean(output_avg_energy(res_path_random),axis=0))


# font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=15) 
# font2 = FontProperties(fname="C:/Windows/Fonts/Times.ttf", size=15)

plt.plot(data_size_lst, energy_ddpg, marker='o', label ='ddpg')
plt.plot(data_size_lst, energy_random, marker='*', label ='random')
# plt.ylabel("奖励",fontproperties=font)
# plt.xlabel("片段",fontproperties=font)

plt.ylabel('energy consumption')
plt.xlabel("packet size")

plt.grid(linestyle=':')

# plt.legend(prop=font)

plt.legend()
plt.show()
