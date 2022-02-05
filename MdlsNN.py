import numpy as np
import PyMieScatt as ps

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class DlsTensorData:
    '''transfer dls_data from numpy array to torch tensor'''
    def __init__(self, dls_data):
        self.tau = torch.from_numpy(dls_data.tau).float()
        self.g1 = torch.from_numpy(dls_data.g1).float()
        self.g1square = torch.from_numpy(dls_data.g1square).float()
        self.intensity = dls_data.intensity
        self.angle = torch.tensor(dls_data.angle).int()
        self.theta = torch.tensor(dls_data.theta).float()


class DlsTensorDataSet:
    '''transfer dls_data_Set from numpy array to torch tensor'''
    def __init__(self, dls_data_set):
        self.tau = torch.from_numpy(dls_data_set.tau).float()
        self.wavelength = torch.tensor(dls_data_set.wavelength).float()
        self.temperature = torch.tensor(dls_data_set.temperature).float()
        self.viscosity = torch.tensor(dls_data_set.viscosity).float()
        self.RI_liquid = torch.tensor(dls_data_set.RI_liquid).float()
        self.RI_particle_complex = dls_data_set.RI_particle_complex
        self.dls_tensor_data_list = [DlsTensorData(dls_data) for dls_data in dls_data_set.dls_data_list]
        

class MdlsDataset(Dataset):
    def __init__(self, dls_tensor_data_set):
        theta_list = [data.theta for data in dls_tensor_data_set.dls_tensor_data_list]
        g1_exp_list = [data.g1 for data in dls_tensor_data_set.dls_tensor_data_list]
        g1square_exp_list = [data.g1square for data in dls_tensor_data_set.dls_tensor_data_list]
        self.theta = torch.stack(theta_list)
        self.g1_exp = torch.stack(g1_exp_list)
        self.g1square_exp = torch.stack(g1square_exp_list)

    def __getitem__(self, index):
        return self.theta[index], self.g1square_exp[index]

    def __len__(self):
        return len(self.theta)


class MdlsModel(nn.Module):

    def __init__(self, dls_tensor_data_set, d, N_init=None, dev='cpu'):
        super().__init__()
        self.d = torch.from_numpy(d).float().to(dev)
        try:
            N_param = self.genNparam(N_init)
            N_param = torch.tensor(N_param).float().to(dev)
        except:
            N_param = torch.rand(d.size)
        self.N_param = nn.Parameter(N_param)
        self.kb = torch.tensor(1.38064852e-23).to(dev)
        self.tau = dls_tensor_data_set.tau.to(dev)
        self.wavelength = dls_tensor_data_set.wavelength.to(dev)
        self.temperature = dls_tensor_data_set.temperature.to(dev)
        self.viscosity = dls_tensor_data_set.viscosity.to(dev)
        self.RI_liquid = dls_tensor_data_set.RI_liquid.to(dev)
        self.RI_particle_complex = dls_tensor_data_set.RI_particle_complex  # python complex
        self.all_theta = torch.stack([data.theta for data in dls_tensor_data_set.dls_tensor_data_list]).to(dev)
        self.all_angle = torch.round(self.all_theta/torch.pi*180).int().to(dev)

        self.dev = dev

        # 事先将完整的散射强度全部计算出来，这样就可以避免训练的时候计算，因为PyMieScatt库只能用CPU算
        self.all_I = self.calcCompleteMieScattringIntensityTensor().to(dev)

    def genN(self):
        # 由于两个原因：1.N只能为非负数但是pytorch不支持限制参数范围；2.不同粒径的N差别可能相当大
        # 所以我们将作为参数的N_param与实际参与计算的N分开，由这个函数来生成实际的N。
        # 这也方便随时调整这个生成函数，而不需要到处改动其他地方
        #N = torch.abs(self.N_param**3)
        N = torch.abs(self.N_param)**3
        return N/torch.sum(N)
        # exp效果很差
        # abs效果也不好
        # 目前三次的效果是最好的，收敛速度快准确度又高容易收敛到真值
        # 二次效果也不错，就是收敛慢一点；四次似乎准确度会差一点
    def genNparam(self, N):
        # 如果输入了N的初始值，则应先转换为N_param
        # 应该与 self.genN() 方法同时修改
        return N**(1/3)

    def forward(self, theta):
        #N = torch.abs(self.N)
        #N = N / torch.sum(N)
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
        w = self.wavelength.item()
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
        '''
        # 这个方法可以用，在GPU上也没问题，但是会更慢
        l = []
        for a in angle:
            l.append(
                (all_angle==a.item()).nonzero().item()
            )
        index = torch.tensor(l).to('cuda')
        '''
        return self.all_I.index_select(0, index)

    def getNumberDistribution(self, to_numpy=True):
        if to_numpy:
            return self.d.cpu().detach().numpy(), self.genN().cpu().detach().numpy()
        else:
            return self.d, self.genN()

    def getIntensityDistribution(self, to_numpy=True):
        theta = self.all_theta.min()  # 默认使用最小角度
        angle = self.all_angle.min()
        d, N = self.d, self.genN()
        I = self.getMieScatteringIntensityTensor(torch.tensor([theta]).to(self.dev))
        G = N*I/torch.sum(N*I)
        G = G.flatten()
        if to_numpy:
            return d.cpu().detach().numpy(), G.cpu().detach().numpy().flatten()
        else:
            return d, G

    def getG(self):
        theta = self.all_theta.min()  # 默认使用最小角度
        I = self.getMieScatteringIntensityTensor(torch.tensor([theta]).to(self.dev))
        N = self.genN()
        G = N*I/torch.sum(N*I)
        G = G.flatten()
        return G
        
    def getG1square(self, to_numpy=True):
        g1square = self.forward(self.all_theta)
        if to_numpy:
            return self.tau.cpu().detach().numpy(), g1square.cpu().detach().numpy()
        else:
            return self.tau, g1square
    

if __name__ == '__main__':
    from DataUtils import genD, simulateDlsData, genDiameterNumDistribution, calcIntensityDistribution, DlsData, DlsDataSet
    from PlotUtils import MdlsNNLogger
    from MdlsNNLS import mdlsNNLS
    import matplotlib.pyplot as plt
    import visdom
    from tqdm import tqdm
    

    epoch_num = 500000
    batch_size = 3
    dev = "cpu"

    param_dict = {
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

    #d, N = [10, 300], [2e10, 5000]  # 大粒子为主的体系 （使用g1效果不太好，即使loss还挺小的）
    #d, N = [10, 300], [2e10, 500]  # 大粒子为主的体系 （使用g1效果不太好，即使loss还挺小的）
    #d, N = [10, 300], [2e10, 1]  # 小粒子为主的体系 （使用g1效果还不错）
    #d, N = [8, 70, 300], [1e13, 1e8, 1e3]  # 中等粒子为主的三峰体系 （使用g1效果还不错）
    d, N = [8, 70, 300], [1e12, 1e8, 1e3]  # 中等粒子为主的三峰体系 （使用g1效果一般，即使loss还挺小的）
    #d, N = genDiameterNumDistribution([50, 200], [5, 100], [1, 10])
    plt.subplot(311)
    plt.scatter(d, N)
    plt.xscale('log')
    plt.subplot(312)
    d, G = calcIntensityDistribution(d, N, 633, 1.5875)
    print(d, G)
    plt.scatter(d, G)
    plt.xscale('log')
    plt.subplot(313)
    angle_list = list(range(15, 150, 15))
    dls_data_set = DlsDataSet(mode='sim')
    for angle in angle_list:
        param_dict['angle'] = angle
        dls_data1 = simulateDlsData(d, N, param_dict=param_dict)
        #plt.plot(dls_data1.tau, dls_data1.g1)
        dls_data_set.addDlsData(dls_data1)
    #plt.xscale('log')
    #plt.show()

    d = genD(d_min=0.1, d_max=2000, d_num=50, log_d=True)
    N_init, norm = mdlsNNLS(dls_data_set, d)

    dls_tensor_data_set = DlsTensorDataSet(dls_data_set)
    train_dataset = MdlsDataset(dls_tensor_data_set)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #model = MdlsModel(dls_tensor_data_set, d_max=1000, dev=dev)
    
    model = MdlsModel(dls_tensor_data_set, d, N_init=None, dev=dev)  # 用NNLS的结果做初始值效果非常差
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(dev)
    epoch_list, loss_list = [], []
    logger = MdlsNNLogger(epoch_num, dls_tensor_data_set, model, environment='MdlsNN')
    print('===== begin MdlsNN training =====')
    for epoch in tqdm(range(epoch_num)):
        for xb, yb in train_dataloader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            epoch_list.append(epoch)
            loss_list.append(loss.item())
            #print('{}\t{}'.format(epoch, loss.item()))
            logger.log(epoch, loss.item())

    plt.subplot(221)
    plt.plot(epoch_list, loss_list)
    plt.yscale('log')
    plt.subplot(222)
    d, N = model.getNumberDistribution()
    d, G = model.getIntensityDistribution()
    np.savetxt('result_distribution.txt', np.vstack((d, N, G)).T, delimiter='\t', header='d\tN\tG')
    #print(d, N)
    plt.plot(d, G)
    plt.xscale('log')
    #np.savetxt('test_training_loss.txt', np.vstack([np.array(epoch_list), np.array(loss_list)]).T, delimiter='\t')
    plt.subplot(212)
    for dlsdata in dls_data_set.dls_data_list:
        plt.plot(dlsdata.tau, dlsdata.g1, '.')
    tau, g1square_all = model.getG1square()
    for g1square in g1square_all:
        plt.plot(tau, g1square, 'k')
    plt.xscale('log')
    plt.savefig('result_plot.jpg', dpi=600)
    plt.show()