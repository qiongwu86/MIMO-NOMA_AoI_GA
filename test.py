#!/usr/bin/python
# -*- coding:utf-8 -*-
from ddpg_lib import *
from multiprocessing import Process
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# def talk(name):
#     print('%s is talking'%name)
#     time.sleep(random.randint(1,3))
#     print('%s is talkend'%name)

# if __name__ == '__main__':
#     p1 = Process(target=talk,args=('Alex',))
#     p2 = Process(target=talk, args=('egon',))
#     p3 = Process(target=talk, args=('lidong',))
#     p4 = Process(target=talk, args=('zhengheng',))

#     p_list_1 = [p1,p2]
#     p_list_2 = [p3,p4]

#     for p in p_list_1:
#         p.start()

#     # 主进程只有在各个子进程执行完毕之后,才会向下执行.
#     for p in p_list_1:
#         p.join()

#     print ('processing')

#     for p in p_list_2:
#         p.start()

#     for p in p_list_2:
#         p.join()

#     print ('finish')


# a = [1,2,3]
# b = [4,5,6]
# c = [7,8,9]

# for d,e,f in zip(a,b,c):
#     print (d,e,f)


# np.random.seed(1)

# print(np.random.uniform(50, 100, [1,2]))
# print(np.random.uniform(50, 100, [1,3]))
# print(np.random.uniform(50, 100, [1,4]))
# print(np.random.uniform(50, 100, [1,5]))

# print (random.uniform(0,1))
# class OU(object):
#     """docstring for OU"""
#     def __init__(self, sig):
#         self.sig =sig
#         self.a = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1),sigma=self.sig)         

#     def function(self,sig):
#         self.sig *= 0.9
#         return self.a()

# lst = []
# sig = 0.2
# n = 400000
# for x in range(0,n):
#     sig *= 0.99997
#     lst.append(OrnsteinUhlenbeckActionNoise(mu=np.zeros(1),sigma=sig)())

# plt.plot(range(n), lst)

# plt.grid(linestyle=':')
# plt.legend()
# plt.show()

# a = [[1,0,1,1],[0,1,0,1]]
# print (np.sum(a))
# print (len(a))

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 准备数据
x = np.linspace(-0.1*np.pi, 2*np.pi, 30)
y_1 = np.sinc(x)+0.7
y_2 = np.tanh(x)
y_3 = np.exp(-np.sinc(x))


# 绘图
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(x, y_1, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C0')

ax.plot(x, y_2, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C3')

ax.plot(x, y_3, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C2')

ax.legend(labels=["y_1", "y_2","y_3"], ncol=3)

# 嵌入绘制局部放大图的坐标系
axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
                   bbox_to_anchor=(0.5, 0.1, 1, 1),
                   bbox_transform=ax.transAxes)

# 在子坐标系中绘制原始数据
axins.plot(x, y_1, color='k', linestyle=':', linewidth=1,
            marker='o', markersize=5,
            markeredgecolor='black', markerfacecolor='C0')

axins.plot(x, y_2, color='k', linestyle=':', linewidth=1,
            marker='o', markersize=5,
            markeredgecolor='black', markerfacecolor='C3')

axins.plot(x, y_3, color='k', linestyle=':', linewidth=1,
            marker='o', markersize=5,
            markeredgecolor='black', markerfacecolor='C2')

# 设置放大区间
zone_left = 11
zone_right = 12

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.5 # x轴显示范围的扩展比例
y_ratio = 0.5 # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

# Y轴的显示范围
y = np.hstack((y_1[zone_left:zone_right], y_2[zone_left:zone_right], y_3[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

# 建立父坐标系与子坐标系的连接线
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

# 显示
plt.show()