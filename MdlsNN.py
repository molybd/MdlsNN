from urllib.response import addinfo
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import PyMieScatt as ps
import copy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import DataUtils
from DataUtils import DlsData, MdlsData
from PlotUtils import MdlsNNLogger

################# constants ###############
'常数'
kb = 1.38064852e-23
###########################################


class DlsTensorData:
    def __init__(self, dlsdata:DlsData) -> None:
        self.tau = torch.from_numpy(dlsdata.tau).float()
        self.g1 = torch.from_numpy(dlsdata.g1).float()
        self.g1square = torch.from_numpy(dlsdata.g1square).float()
        self.intensity = dlsdata.intensity
        self.angle = torch.tensor(dlsdata.angle).int()
        self.theta = torch.tensor(dlsdata.theta).float()
        self.params = dlsdata.params

class MdlsTensorData:
    def __init__(self, mdlsdata:MdlsData=None) -> None:
        if mdlsdata == None:
            self.data = {}
        else:
            self.data = {}
            for dlsdata in mdlsdata.data.values():
                self.addDlsData(dlsdata)
            self.params = mdlsdata.params

    def addDlsData(self, dlsdata:DlsData) -> None:
        self.data[dlsdata.angle] = DlsTensorData(dlsdata)
        self.params = dlsdata.params

class MdlsDataset(Dataset):
    '''for pytorch dataloader use
    use g1**2 for training
    '''
    def __init__(self, mdls_tensordata:MdlsTensorData, dev:str='cpu') -> None:
        # 注意两个输出必须具有相同的顺序，一一对应
        super().__init__()
        self.mdls_tensordata = mdls_tensordata
        angle_list = list(mdls_tensordata.data.keys())
        angle_tensor_list = [torch.tensor(angle).int() for angle in angle_list]
        g1square_list = [mdls_tensordata.data[angle].g1square for angle in angle_list]
        self.angle = torch.stack(angle_tensor_list).to(dev)
        self.g1square = torch.stack(g1square_list).to(dev)

    def __getitem__(self, index):
        return self.angle[index], self.g1square[index]

    def __len__(self):
        return len(self.angle)


class MdlsNNModel(nn.Module):
    def __init__(self, mdls_tensordata:MdlsTensorData, d:ndarray, N_init:ndarray=None, dev:str='cpu') -> None:
        super().__init__()
        self.dev = dev
        self.mdls_tensordata = mdls_tensordata
        self.d = torch.from_numpy(d).float().to(dev)
        try:
            param_N = self.genParamN(N_init)
            param_N = torch.tensor(param_N).float().to(dev)
        except:
            param_N = torch.rand(d.size)
        self.param_N = nn.Parameter(param_N)  # 模型参数
        
        self.scatt_int = self.genMieScattIntDict()

        self.ri_liquid = torch.tensor(self.mdls_tensordata.params.ri_liquid).to(self.dev)
        self.wavelength = torch.tensor(self.mdls_tensordata.params.wavelength).to(self.dev)
        self.temperature = torch.tensor(self.mdls_tensordata.params.temperature).to(self.dev)
        self.viscosity = torch.tensor(self.mdls_tensordata.params.viscosity).to(self.dev)
        tau = list(self.mdls_tensordata.data.values())[0].tau
        self.tau = tau.to(self.dev)
        self.angle = torch.tensor(list(self.mdls_tensordata.data.keys())).to(dev)
        
        # 测试用
        '''
        self.kb = kb
        self.RI_particle_complex = complex(
            self.mdls_tensordata.params.ri_particle_real, 
            self.mdls_tensordata.params.ri_particle_img
            )
        self.RI_liquid = self.mdls_tensordata.params.ri_liquid
        self.all_theta = self.angle/180*torch.pi
        self.all_angle = torch.round(self.all_theta/torch.pi*180).int().to(dev)
        self.all_I = self.calcCompleteMieScattringIntensityTensor().to(dev)
        '''

    def genN(self):
        '''由于两个原因：1.N只能为非负数但是pytorch不支持限制参数范围；2.不同粒径的N差别可能相当大
        所以我们将作为模型参数的N_param与实际参与计算的N分开，由这个函数来生成实际参与神经网络训练的N。
        这也方便随时调整这个生成函数，而不需要到处改动其他地方
        '''
        # exp效果很差
        # abs效果也不好
        # 目前三次的效果是最好的，收敛速度快准确度又高容易收敛到真值
        # 二次效果也不错，就是收敛慢一点；四次似乎准确度会差一点
        N = torch.abs(self.param_N)**3
        return N/torch.sum(N)
    def genParamN(self, N):
        # 如果输入了N的初始值，则应先转换为paramN
        # 应该与 self.genN() 方法同时修改
        return N**(1/3)
    
    def genMieScattIntDict(self) -> dict:
        wavelength = self.mdls_tensordata.params.wavelength
        ri_particle_complex = complex(
            self.mdls_tensordata.params.ri_particle_real, 
            self.mdls_tensordata.params.ri_particle_img
            )
        d = self.d.cpu().detach().numpy()
        angle_list = list(self.mdls_tensordata.data.keys())
        if 90 not in angle_list:
            angle_list.append(90)  # 增加90度，为了输出90度的光强数据
        scatt_int_dict = {}
        for angle in angle_list:
            theta = angle/180*torch.pi
            Id = []
            for di in d:
                Id.append(DataUtils.mieScattInt(
                    theta, di, wavelength, ri_particle_complex
                ))
            scatt_int_dict[angle] = torch.tensor(Id).float().to(self.dev)
        return scatt_int_dict

    def forward(self, angle):
        theta = angle/180*np.pi  # (n_angle,)
        q = (4*torch.pi*self.ri_liquid*torch.sin(theta/2)) / self.wavelength  # (n_angle,)
        Diffuse = (kb*self.temperature) / (3*torch.pi*self.viscosity*self.d)  # (n_d,)
        Gamma = 1e24 * torch.einsum('i,j->ij', q**2, Diffuse) # (n_angle, n_d) make unit μsec^-1
        g1i = torch.exp(
            -1*torch.einsum('ij,k->ijk', Gamma, self.tau)
        )  # (n_angle, n_d, n_tau)
        Id = torch.stack(
            [self.scatt_int[a.item()] for a in angle]
        )  # (n_angle, n_d)
        N = self.genN()  # (n_d,)
        NId = torch.einsum('ij,j->ij', Id, N)  # (n_angle, n_d)
        sumNId = torch.einsum('ij->i', NId)  # (n_angle,)
        G = torch.einsum('ij,i->ij', NId, 1/sumNId)  # (n_angle, n_d)
        g1 = torch.einsum('ij,ijk->ik', G, g1i) # (n_angle, n_tau)
        return g1**2

    '''
    def forward_old(self, angle):
        #N = torch.abs(self.N)
        #N = N / torch.sum(N)
        theta = angle/180*torch.pi
        g1 = self.simulateG1(
            self.d,
            self.genN(),
            self.tau,
            theta,
            self.wavelength,
            self.temperature,
            self.viscosity,
            self.RI_liquid
        )
        g1square = torch.sign(g1) * g1**2
        return g1square

    def simulateG1(self, d, N, tau, theta, wavelength, temperature, viscosity, RI_liquid):
        q = (4*torch.pi*RI_liquid*torch.sin(theta/2)) / wavelength       # shape == (n_theta,)
        Diffuse = (self.kb*temperature) / (3*torch.pi*viscosity*d)             # shape == (n_d,)
        #Gamma = Diffuse * q**2 * 1e24  # make unit μm^-1               # shape == (n_d,)
        Gamma = torch.einsum('i,j->ij', q**2, Diffuse) * 1e24        # shape == (n_theta, n_d)

        I = self.getMieScatteringIntensityTensor(theta)           # shape == (n_theta, n_d)

        #I_d = [self.calcMieScatteringIntensity(theta, di, wavelength, RI_particle_complex) for di in d]
        #I_d = torch.tensor(I_d).float()                                  # shape == (n_d,)

        I_N = torch.einsum('ij,j->ij', I, N)            # shape == (n_theta, n_d)
        I_N_sum = torch.sum(I_N, dim=1)                 # shape == (n_theta,)
        r_I_N_sum = 1/I_N_sum                           # shape == (n_theta,)
        G = torch.einsum('ij,i->ij', I_N, r_I_N_sum)      # shape == (n_theta, n_d)
        exp = torch.exp(-1*torch.einsum('ij,k->ijk', Gamma, tau))  # shape == (n_theta, n_d, n_tau)
        g1 = torch.einsum('ij,ijk->ik', G, exp)                   # shape == (n_theta, n_tau)
        return g1

    def calcCompleteMieScattringIntensityTensor(self):
        w = self.wavelength
        m = self.RI_particle_complex
        I = []
        for ti in self.all_theta:
            ti = ti.item()
            mu = np.cos(ti)
            temp = []
            for di in self.d:
                di = di.item()
                x = np.pi*di/w
                S1, S2 = ps.MieS1S2(m, x, mu)
                temp.append(np.abs(S1)**2)
            I.append(temp)
        I = torch.tensor(I).float()       
        return I    # shape == (n_theta, n_d)

    def getMieScatteringIntensityTensor(self, theta):
        # 用纯pytorch方法获得某些角度的散射强度，从而可以在GPU上算
        # 从事先计算好的 self.I 中获得值
        angle = torch.round(theta/torch.pi*180).int()  # shape == (n,)
        all_angle = self.all_angle                 # shape == (all,)
        
        # 为了实现纯张量操作
        angle = angle.view(angle.size(dim=0), 1)   # (n, 1)
        angle = angle.expand(angle.size(dim=0), all_angle.size(dim=0))  # (n, all)
        all_angle = all_angle.expand(angle.size(dim=0), all_angle.size(dim=0))  # (n, all)
        index = angle - all_angle
        index = (index==0).nonzero()[:,1].flatten()
        
        return self.all_I.index_select(0, index)
    '''
    def getD(self, to_numpy=True):
        if to_numpy:
            return self.d.cpu().detach().numpy()
        else:
            return self.d

    def getN(self, to_numpy=True):
        if to_numpy:
            return self.genN().cpu().detach().numpy()
        else:
            return self.genN()

    def getG(self, to_numpy=True):
        angle = 90  # 默认使用90度
        N = self.getN(to_numpy=False)
        I = self.scatt_int[angle]
        G = N*I / torch.sum(N*I)
        G = G.flatten()
        if to_numpy:
            return G.cpu().detach().numpy()
        else:
            return G

    def getG1square(self, to_numpy=True):
        g1square = self.forward(self.angle)
        if to_numpy:
            return g1square.cpu().detach().numpy()
        else:
            return g1square



class Train:
    default_params = {
        'batch_size': 3,
        'shuffle': True,
        'epoch_num': 500000,
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'loss': 'MSELoss'
    }
    def __init__(self, mdlsdata:MdlsData, train_params:dict=None) -> None:
        self.mdlsdata = mdlsdata
        self.mdls_tensordata = MdlsTensorData(mdlsdata)
        if train_params == None:
            train_params = self.default_params
        self.train_params = copy.deepcopy(train_params)
        #self.dataloader = DataLoader()

    def MSELossWithPenalty(self, y_pred, yb, N, G, weight_N=0, weight_G=0):
        mseloss = torch.mean((yb-y_pred)**2)
        panelty_N = weight_N * torch.mean((N[2:]-2*N[1:-1]+N[:-2])**2)
        panelty_G = weight_G * torch.mean((G[2:]-2*G[1:-1]+G[:-2])**2)
        return mseloss + panelty_N + panelty_G

    def train(self, d:ndarray, train_params:dict=None, dev='cpu', visdom_log=False, env_name='MdlsNN'):
        if train_params != None:
            for key, value in train_params:
                self.train_params[key] = value
        
        dataset = MdlsDataset(self.mdls_tensordata, dev=dev)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.train_params['batch_size'],
            shuffle=self.train_params['shuffle']
            )
        model = MdlsNNModel(self.mdls_tensordata, d, dev=dev)
        mse_criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.train_params['learning_rate']
            )
        model.to(dev)
        
        epoch_list, loss_list = [], []
        epoch_num = self.train_params['epoch_num']
        if visdom_log:
            logger = MdlsNNLogger(epoch_num, self.mdls_tensordata, model, environment=env_name)
        for epoch in tqdm(range(epoch_num), desc=env_name, unit='epoch'):
        #for epoch in range(epoch_num):
            for xb, yb in dataloader:
                y_pred = model(xb)
                loss = mse_criterion(y_pred, yb)
                #N = model.getN(to_numpy=False)
                #G = model.getG(to_numpy=False)
                #loss = self.MSELossWithPenalty(y_pred, yb, N, G, weight_N=1e-6, weight_G=1e-4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                if visdom_log:
                    logger.log(epoch, loss.item())
                    epoch_list.append(epoch)
                    loss_list.append(loss.item())
                #print(epoch, loss.item())
        self.model = model
        d = self.model.getD(to_numpy=True)
        N = self.model.getN(to_numpy=True)
        G = self.model.getG(to_numpy=True)

        fit_g1square = {}
        for angle, dlsdata in self.mdlsdata.data.items():
            g1square = DataUtils.calcG1(dlsdata.tau, d, N, angle, dlsdata.params)**2
            fit_g1square[angle] = g1square.tolist()

        self.model = model
        self.train_params['d'] = d.tolist()
        self.train_params['dev'] = dev
        self.result = {
            'epoch': epoch_list,
            'loss': loss_list,
            'd': d.tolist(),
            'N': N.tolist(),
            'G': G.tolist(),
            'fit_g1square': fit_g1square
        }

    def toDict(self) -> dict:
        dic = {
            'train_params': self.train_params,
            'mdlsdata': self.mdlsdata.toDict(),
            'result': self.result
        }
        return dic
    


if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    import json
    
    test_list = [
        'data\simdata_continuous_3.json',
        'data\simdata_continuous_2.json',
        'data\simdata_unimodal_2.json',
        'data\simdata_bimodal_5.json',
        'data\simdata_bimodal_1.json',
        'data\simdata_bimodal_6.json',
        'data\simdata_trimodal_1.json',
        'data\simdata_trimodal_3.json',
    ]
    
    for filename in test_list:
        with open(filename, 'r') as f:
            mdlsdata = MdlsData(json.load(f))
    
        train = Train(mdlsdata)
        d = np.logspace(0, np.log10(1e4), num=50)
        env_name = filename.split('\\')[-1].split('.')[0]
        train.train(d, dev='cuda', visdom_log=True, env_name=env_name)

        result_name = env_name + '_result-MESLoss-dmax=1e4'
        with open('data/{}.json'.format(result_name), 'w') as f:
            json.dump(train.toDict(), f, indent=4)


    