import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle

import DataUtils
import MdlsNN
from PlotUtils import MdlsNNLogger
from FitExperiment import *


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #d_N_list = [
    #    ([63], [1]),                         #0
    #    ([10, 300], [2e10, 1]),              #1 小粒子为主的体系
    #    ([10, 300], [2e10, 3]),              #2 小粒子为主的体系
    #    ([10, 300], [2e10, 10]),             #3 小粒子为主的体系
    #    ([10, 300], [2e10, 100]),            #4 大小粒子差不太多的体系
    #    ([10, 300], [2e10, 5000]),           #5 大粒子为主的体系
    #    ([10, 300], [2e10, 500]),            #6 大粒子为主的体系
    #    ([10, 70, 300], [2e10, 2e4, 1]),     #7 小粒子为主的三峰体系
    #    ([10, 70, 300], [5e11, 1e8, 3e3]),   #8 中粒子为主的三峰体系
    #    ([10, 70, 300], [3e12, 1e7, 1e5]),   #9 大粒子为主的三峰体系
    #    ([10, 70, 300], [1.5e13, 1e8, 4e4]), #10 三种粒子差不多的三峰体系
    #    ([10, 800], [1e9, 1]),               #11 粒径差距稍大的体系，小粒子为主
    #    ([10, 800], [1e9, 3e1]),             #12 粒径差距稍大的体系，大小粒子差不多
    #    ([10, 800], [1e9, 1e3])              #13 粒径差距稍大的体系，大粒子为主
    #    ([10, 50], [5e5, 1]),              #14 小粒子为主的体系
    #    ([10, 50], [2e5, 10]),              #15 差不多的体系
    #    ([10, 50], [2e5, 300]),             #16 大粒子为主的体系
    #    ([200, 400], [100, 1]),              #17 小粒子为主的体系
    #    ([200, 400], [100, 10]),              #18 差不多的体系
    #    ([200, 400], [100, 150]),             #19 大粒子为主的体系.
    #    ([500, 800], [100, 1]),              #20 小粒子为主的体系
    #    ([500, 800], [100, 10]),              #21 差不多的体系
    #    ([500, 800], [100, 200]),             #22 大粒子为主的体系
    #    ([200, 400], [100, 10]),              #23 大小粒子差不多的体系
    #    ([200, 400], [100, 150]),              #24 大粒子为主的体系
    #    ([500, 800], [100, 10]),              #25 差不多的体系
    #    ([500, 800], [100, 10]),              #26 差不多的体系
    #                                          #27
    #]

    epoch_num = 500000
    batch_size = 3
    dev = "cpu"
    angle_list = list(range(15, 150, 15))

    #d_N_list = []
    #d_N_list.append(DataUtils.genDiameterNumDistribution(200, 20, 1, d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution(400, 150, 1, d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([200,400], [10,50], [100,10], d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([50,500], [20,50], [1e6,1], d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([200,500], [20,200], [2e2,1], d_min=1, d_max=1000, d_num=1000, log_d=False))
#
    #for i in range(len(d_N_list)):
    #    exp_index = i + 28
    #    print('======= begin experiment {} ======'.format(exp_index))
    #    print('simulated intensity distribution:')
    #    d, N = d_N_list[i]
    #    d_temp, G = DataUtils.calcIntensityDistribution(d, N, 633, 1.5875)
    #    print(d_temp, G)
    #    dls_data_set = simMultiangleDlsDataSet(d, N, angle_list, param_dict={'tau_min':5, 'tau_max':1e6})
    #    d = DataUtils.genD(d_min=1, d_max=100000, d_num=50)
    #    experiment = Experiment(dls_data_set)
    #    experiment.doMdlsNN(d, epoch_num, batch_size, dev=dev, visdom_environment='exp_{}'.format(exp_index))
    #    saveExperiment(experiment, './fitting experiments/MdlsNN-fit_sim-data_{}.pickle'.format(exp_index))
    #
    #
    #d_N_list = []
    #d_N_list.append(DataUtils.genDiameterNumDistribution(200, 20, 1, d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution(400, 150, 1, d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([200,400], [10,50], [100,10], d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([50,500], [20,50], [1e6,1], d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([200,500], [20,200], [2e2,1], d_min=1, d_max=1000, d_num=1000, log_d=False))
    #
    #for i in range(len(d_N_list)):
    #    exp_index = i + 33
    #    print('======= begin experiment {} ======'.format(exp_index))
    #    print('simulated intensity distribution:')
    #    d, N = d_N_list[i]
    #    d_temp, G = DataUtils.calcIntensityDistribution(d, N, 633, 1.5875)
    #    print(d_temp, G)
    #    dls_data_set = simMultiangleDlsDataSet(d, N, angle_list, param_dict={'tau_min':5, 'tau_max':1e6})
    #    d = DataUtils.genD(d_min=1, d_max=1000, d_num=50, log_d=False)
    #    experiment = Experiment(dls_data_set)
    #    experiment.doMdlsNN(d, epoch_num, batch_size, dev=dev, visdom_environment='exp_{}'.format(exp_index))
    #    saveExperiment(experiment, './fitting experiments/MdlsNN-fit_sim-data_{}.pickle'.format(exp_index))
    #
    #d_N_list = []
    #d_N_list.append(DataUtils.genDiameterNumDistribution(200, 20, 1, d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution(400, 150, 1, d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([200,400], [10,50], [100,10], d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([50,500], [20,50], [1e6,1], d_min=1, d_max=1000, d_num=1000, log_d=False))
    #d_N_list.append(DataUtils.genDiameterNumDistribution([200,500], [20,200], [2e2,1], d_min=1, d_max=1000, d_num=1000, log_d=False))
    #
    #for i in range(len(d_N_list)):
    #    exp_index = i + 38
    #    print('======= begin experiment {} ======'.format(exp_index))
    #    print('simulated intensity distribution:')
    #    d, N = d_N_list[i]
    #    d_temp, G = DataUtils.calcIntensityDistribution(d, N, 633, 1.5875)
    #    print(d_temp, G)
    #    dls_data_set = simMultiangleDlsDataSet(d, N, angle_list, param_dict={'tau_min':5, 'tau_max':1e6})
    #    d = DataUtils.genD(d_min=1, d_max=2000, d_num=50, log_d=False)
    #    experiment = Experiment(dls_data_set)
    #    experiment.doMdlsNN(d, epoch_num, batch_size, dev=dev, visdom_environment='exp_{}'.format(exp_index))
    #    saveExperiment(experiment, './fitting experiments/MdlsNN-fit_sim-data_{}.pickle'.format(exp_index))

    def plot_d_G(d_N_list):
        for i in range(len(d_N_list)):
            plt.plot(*DataUtils.calcIntensityDistribution(*d_N_list[i], 633, 1.5875), label=str(i))
        plt.legend()
        plt.savefig('test.png')

    d_N_list = []
    d_N_list.append(DataUtils.genDiameterNumDistribution([100,400], [10,50], [1e4,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([100,400], [10,50], [1e2,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([100,400], [10,50], [1e0,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([400,500], [10,100], [5,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([400,500], [10,100], [1,2], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([400,500], [10,100], [1,100], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 20, 200], [1e2,1,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 20, 200], [1e1,10,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 20, 200], [1e1,1,5], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 200, 20], [1e2,10,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    
    for i in range(len(d_N_list)):
        exp_index = i + 43
        print('======= begin experiment {} ======'.format(exp_index))
        #print('simulated intensity distribution:')
        d, N = d_N_list[i]
        d_temp, G = DataUtils.calcIntensityDistribution(d, N, 633, 1.5875)
        #print(d_temp, G)
        dls_data_set = simMultiangleDlsDataSet(d, N, angle_list, param_dict={'tau_min':0.1, 'tau_max':1e6})
        d = DataUtils.genD(d_min=1, d_max=2000, d_num=50, log_d=False)
        experiment = Experiment(dls_data_set)
        experiment.doMdlsNN(d, epoch_num, batch_size, dev=dev, visdom_environment='exp_{}'.format(exp_index))
        saveExperiment(experiment, './fitting experiments/MdlsNN-fit_sim-data_{}.pickle'.format(exp_index))


    d_N_list = []
    d_N_list.append(DataUtils.genDiameterNumDistribution([100,400], [10,50], [1e4,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([100,400], [10,50], [1e2,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([100,400], [10,50], [1e0,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([400,500], [10,100], [5,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([400,500], [10,100], [1,2], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([400,500], [10,100], [1,100], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 20, 200], [1e2,1,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 20, 200], [1e1,10,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 20, 200], [1e1,1,5], d_min=1, d_max=2000, d_num=1000, log_d=False))
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 200, 20], [1e2,10,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    
    for i in range(len(d_N_list)):
        exp_index = i + 53
        print('======= begin experiment {} ======'.format(exp_index))
        #print('simulated intensity distribution:')
        d, N = d_N_list[i]
        d_temp, G = DataUtils.calcIntensityDistribution(d, N, 633, 1.5875)
        #print(d_temp, G)
        dls_data_set = simMultiangleDlsDataSet(d, N, angle_list, param_dict={'tau_min':0.1, 'tau_max':1e6})
        d = DataUtils.genD(d_min=1, d_max=2000, d_num=500, log_d=False)
        experiment = Experiment(dls_data_set)
        experiment.doMdlsNN(d, epoch_num, batch_size, dev=dev, visdom_environment='exp_{}'.format(exp_index))
        saveExperiment(experiment, './fitting experiments/MdlsNN-fit_sim-data_{}.pickle'.format(exp_index))

    