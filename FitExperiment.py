import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle

import DataUtils
import MdlsNN
from PlotUtils import MdlsNNLogger


class Experiment:

    def __init__(self, dls_data_set):
        self.dls_data_set = dls_data_set
        self.mode = dls_data_set.mode
        if self.mode == 'sim':
            self.sim_d = self.dls_data_set.dls_data_list[0].sim_info_dict['d']
            self.sim_N = self.dls_data_set.dls_data_list[0].sim_info_dict['N']
            wavelength = self.dls_data_set.wavelength
            m = self.dls_data_set.RI_particle_complex
            smallest_theta = min([dls_data.theta for dls_data in dls_data_set.dls_data_list])
            self.sim_G = DataUtils.calcIntensityDistribution(self.sim_d, self.sim_N, wavelength, m, theta=smallest_theta)[1]

    def lossFuncWith2ndDerivPanelty(self, N, G, y_pred, y, weight_N=0, weight_G=0):
        mseloss = torch.mean((y-y_pred)**2)
        panelty_N = weight_N * torch.mean((N[2:]-2*N[1:-1]+N[:-2])**2)
        panelty_G = weight_G * torch.mean((G[2:]-2*G[1:-1]+G[:-2])**2)
        return mseloss + panelty_N + panelty_G
        
    def doMdlsNN(self, d, epoch_num, batch_size, dev='cpu', visdom_log=False, visdom_environment='MdlsNN'):
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.dev = dev
        self.loss_func = 'MSELoss'
        self.optimizer = 'Adam'
        self.learning_rate = 0.001
        dls_tensor_data_set = MdlsNN.DlsTensorDataSet(self.dls_data_set)
        train_dataset = MdlsNN.MdlsDataset(dls_tensor_data_set)
        train_dataloader = MdlsNN.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = MdlsNN.MdlsModel(dls_tensor_data_set, d, dev=dev)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.to(dev)
        if visdom_log:
            logger = MdlsNNLogger(self.epoch_num, dls_tensor_data_set, model, diameter_distribution_xtype='linear', environment=visdom_environment)
        #print('===== begin MdlsNN training =====')
        epoch_list, loss_list = [], []
        for epoch in tqdm(range(epoch_num), desc='MdlsNN training process', unit='epoch'):
            for xb, yb in train_dataloader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                pred = model(xb)
                #loss = criterion(pred, yb)
                # new loss with 2nd derivative panelty
                N = model.genN()
                G = model.getG()
                loss = self.lossFuncWith2ndDerivPanelty(N, G, pred, yb, weight_N=1e-10, weight_G=1e-4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                if visdom_log:
                    logger.log(epoch, loss.item())
                epoch_list.append(epoch)
                loss_list.append(loss.item())
                #print(loss.item())
        d, N = model.getNumberDistribution()
        d, G = model.getIntensityDistribution()
        self.result_d = d
        self.result_N = N
        self.result_G = G
        self.record_epoch = np.array(epoch_list)
        self.record_loss = np.array(loss_list)
        tau, g1square = model.getG1square()
        self.result_tau = tau
        self.result_g1square = g1square
        self.model = model
        print('Training finished')
        print('=================================')
        

def saveExperiment(experiment, filename):
    with open(filename, 'wb') as f:
        pickle.dump(experiment, f)

def loadExperiment(filename):
    with open(filename, 'rb') as f:
        experiment = pickle.load(f)
    return experiment

def simMultiangleDlsDataSet(d, N, angle_list, param_dict=None):
    default_param_dict = {
        'tau_min': 0.1,              # microsec
        'tau_max': 1e6,              # microsec
        'tau_num': 200,              
        'angle': 90,                 # degree
        'wavelength': 633,           # nanometer
        'temperature': 298,          # Kelvin
        'viscosity': 0.89,           # cP
        'RI_liquid': 1.331,
        'RI_particle_real': 1.5875,
        'RI_particle_img': 0,
        'baseline': 55225135654
        }
    if param_dict != None:
        for key, value in param_dict.items():
            default_param_dict[key] = value
    param_dict = default_param_dict

    dls_data_set = DataUtils.DlsDataSet(mode='sim')
    for angle in angle_list:
        param_dict['angle'] = angle
        dls_data = DataUtils.simulateDlsData(d, N, param_dict=param_dict)
        dls_data_set.addDlsData(dls_data)
    return dls_data_set



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PlotUtils import plotFitExperimentResult
    

    epoch_num = 500000
    batch_size = 3
    dev = "cpu"
    angle_list = list(range(15, 150, 15))

    
    def plot_d_G(d_N_list):
        for i in range(len(d_N_list)):
            plt.plot(*DataUtils.calcIntensityDistribution(*d_N_list[i], 633, 1.5875), label=str(i))
        plt.legend()
        plt.savefig('test.png')

    d_N_list = []
    d_N_list.append(DataUtils.genDiameterNumDistribution([200, 600,700], [50, 20, 200], [1e2,1,1], d_min=1, d_max=2000, d_num=1000, log_d=False))
    
    for i in range(len(d_N_list)):
        exp_index = i + 63
        print('======= begin experiment {} ======'.format(exp_index))
        #print('simulated intensity distribution:')
        d, N = d_N_list[i]
        d_temp, G = DataUtils.calcIntensityDistribution(d, N, 633, 1.5875)
        #print(d_temp, G)
        dls_data_set = simMultiangleDlsDataSet(d, N, angle_list, param_dict={'tau_min':0.1, 'tau_max':1e6})
        d = DataUtils.genD(d_min=1, d_max=2000, d_num=50, log_d=False)
        experiment = Experiment(dls_data_set)
        experiment.doMdlsNN(d, epoch_num, batch_size, dev=dev, visdom_log=True, visdom_environment='exp_{}'.format(exp_index))
        saveExperiment(experiment, './fitting experiments/MdlsNN-fit_sim-data_panelty-test.pickle')
        plotFitExperimentResult(experiment, figname='./fitting experiments/MdlsNN-fit_sim-data_panelty-test.png', figsize=(15, 7.5), show=False)
        plotFitExperimentResult(experiment, figname='./fitting experiments/MdlsNN-fit_sim-data_panelty-test.svg', figsize=(15, 7.5), show=False)