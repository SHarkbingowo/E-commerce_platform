# write by LU
# TIME:2022/1/20 11:54

import os
from numpy import array
import numpy as np
import matplotlib.pyplot as plt

def draw_hist(data, xlabel):
    a = plt.hist(data,30)

#设置出横坐标
    plt.xlabel('{}出现的频数'.format(xlabel),fontsize=15)
#设置纵坐标的标题
    plt.ylabel('出现的频数',fontsize=15)
#设置整个图片的标题
    plt.title('{}的直方图'.format(xlabel),fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

# #
#     for p,b in zip(a[0],a[1]):#a[0]是每根柱子的长度，为一个列表，a[1]就是bins列表
#
#         plt.text(b, 1.02*p, str(round(p*100,2)),fontsize=8)#前两个参数确定柱子的位置，1.02*p代表在柱子顶稍高一点，第三个参数设定显示数据，fontsize规定字号



