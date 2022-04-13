import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *
from matplotlib.font_manager import FontProperties 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=15) 
font2 = FontProperties(fname="C:/Windows/Fonts/Times.ttf", size=15)
# font2 = {'family' : 'Times',
# 'weight' : 'normal',
# 'size'   : 15,
# }


def moving_average(a, n=1) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def output_avg(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []

    for name in fileList:
    	path = dir_path + name
    	res = np.load(path)
    	temp_rs = np.array(res['arr_0'])
    	avg_rs.append(temp_rs)

# for debug-----------to check each training result 
    # n = 11
    # path = dir_path + fileList[n]
    # print (fileList, fileList[n])
    # res = np.load(path)
    # temp_rs = np.array(res['arr_0'])
    # avg_rs.append(temp_rs)

    avg_rs = moving_average(np.mean(avg_rs, axis=0, keepdims=True)[0],8)
    return avg_rs

def out(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    for name in fileList:
        path = dir_path + name
        res = np.load(path)
        temp_rs = moving_average(np.array(res['arr_0']))
        avg_rs.append(temp_rs)
    return np.array(avg_rs)

user_num_list = [2,3,4,5,6]
marker_lst = ['d','v','x','h','s']

Data_file = 'Data_t_0.5_σ_0.004'

fig, ax = plt.subplots(1, 1)

for user_num in user_num_list:
    res_path = Data_file + '/train/ddpg/varying_usernum/usernum_' + str(user_num) + '/'
    reward = output_avg(res_path)
    ax.plot(range(reward.shape[0]), reward, label =  str(user_num) + ' IoTDs', marker = marker_lst[user_num-2], markevery = 40)

ax.legend(prop=font2)
ax.grid(linestyle=':')
# plt.ylabel("reward")
# plt.xlabel("episode index")
plt.ylabel("奖励",fontproperties=font)
plt.xlabel("片段",fontproperties=font)

# 嵌入绘制局部放大图的坐标系
axins = ax.inset_axes((0.2, 0.2, 0.4, 0.3))

#在子坐标系中绘制原始数据
for user_num in user_num_list:
    res_path = Data_file + '/train/ddpg/varying_usernum/usernum_' + str(user_num) + '/'
    reward = output_avg(res_path)
    axins.plot(range(reward.shape[0]), reward, label =  str(user_num) + ' IoTDs', marker = marker_lst[user_num-2], markevery = 40)

# 设置放大区间
zone_left = 700
zone_right = 770

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.5 # x轴显示范围的扩展比例
y_ratio = 0.5 # y轴显示范围的扩展比例

# X轴的显示范围
x = range(output_avg(Data_file + '/train/ddpg/varying_usernum/usernum_' + '2' + '/').shape[0])
xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

reward_lst = []
# Y轴的显示范围
for user_num in user_num_list:   
    res_path = Data_file + '/train/ddpg/varying_usernum/usernum_' + str(user_num) + '/'
    reward_lst.append(output_avg(res_path))

y = np.hstack((reward_lst[0][zone_left:zone_right],reward_lst[1][zone_left:zone_right],reward_lst[2][zone_left:zone_right],reward_lst[3][zone_left:zone_right],reward_lst[4][zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

# 建立父坐标系与子坐标系的连接线
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec='k', lw=1)

plt.show()