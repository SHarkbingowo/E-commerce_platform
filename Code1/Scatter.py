import matplotlib.pyplot as plt

def draw_Scatter(data,xlabel):
    plt.scatter(x=list(range(len(data))), y=data)
    plt.xlabel('{}的各样本点'.format(xlabel),fontsize=15)
    plt.ylabel('{}的样本点取值'.format((xlabel)),fontsize=15)
    plt.title('{}的散点图'.format(xlabel),fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)



def draw_outlierScatter(data,data_drop,variable,varnum):
    plt.scatter(x=list(data.index), y=data[variable], label='正常点')
    plt.scatter(x=list(data_drop.index), y=data_drop[variable], label='离群点')
    plt.xlabel('{}的各样本点'.format(varnum),fontsize=15)
    plt.ylabel('{}的样本点取值'.format(varnum),fontsize=15)
    plt.title('{}的散点图'.format(varnum),fontsize=15)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)