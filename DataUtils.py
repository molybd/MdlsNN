import string
import numpy as np
from numpy import ndarray
import PyMieScatt as ps
import copy
import os

'''
====== Attention ======
at all place, the units are the regularly used units, c.a.
#param          #unit
tau          microsecond
angle        degree  # must be integer
theta        radian
wavelength   nanometer
viscosity    cP
temperature  Kelvin
diameter     nanometer
kb           J/K
at these units, the calculated Gamma should be multiplied by 1e24 to
become μm^(-1) unit that fits tau. for example, if gamma calculated is
2, then you should make it 2e24, then calculate g1 = exp(-gamma*tau)

=====================================
some abbreviation or code names
# name                  # simple name
diameter                d
number distribution     N
intensity distribution  G
simulate/simulation     sim
experimental            exp
=====================================
'''

################# constants ###############
'常数'
kb = 1.38064852e-23
###########################################



class Params:
    '''
    储存和具体测试数据无关的参数
    '''
    default = {
        'wavelength': 633,           # nanometer
        'temperature': 298,          # Kelvin
        'viscosity': 0.89,           # cP
        'ri_liquid': 1.331,
        'ri_particle_real': 1.5875,
        'ri_particle_img': 0
    }
    def __init__(self, params_dict:dict=None) -> None:
        if params_dict == None:
            params_dict = copy.deepcopy(self.default)
        self._update(params_dict)

    def setParam(self, name:str, value:float):
        params_dict = self.genDict()
        params_dict[name] = value
        self._update(params_dict)

    def _update(self, params_dict:dict):
        self.wavelength = params_dict['wavelength']
        self.temperature = params_dict['temperature']
        self.viscosity = params_dict['viscosity']
        self.ri_liquid = params_dict['ri_liquid']
        self.ri_particle_real = params_dict['ri_particle_real']
        self.ri_particle_img = params_dict['ri_particle_img']

    def toDict(self) -> dict:
        dic = copy.deepcopy(self.__dict__)  # 深复制一个，否则更改__dict__会直接改变对象参数的值
        return dic



################### calculation related functions ###########################
'''与计算过程相关的函数'''

def calcQ(theta:float, wavelength:float, ri_liquid:float) -> float:
    return (4*np.pi*ri_liquid*np.sin(theta/2)) / (wavelength)

def calcDiffuse(d:ndarray, temperature:float, viscosity:float) -> ndarray:
    return (kb*temperature) / (3*np.pi*viscosity*d)

def calcGamma(Diffuse:ndarray, q:float) -> ndarray:
    return Diffuse * q**2 * 1e24  # make unit μsec^-1

def calcG1(tau:ndarray, d:ndarray, N:ndarray, angle:float, params:Params) -> ndarray:
    '''
    模拟多分散样品在单个角度的场自相关函数g1
    '''
    theta = angle/180*np.pi
    ri_particle_complex = complex(params.ri_particle_real, params.ri_particle_img)

    q = calcQ(theta, params.wavelength, params.ri_liquid)
    D = calcDiffuse(d, params.temperature, params.viscosity)
    Gamma = calcGamma(D, q)  # shape=(n_d,)
    g1i = np.exp(-1*np.einsum('i,j->ij', Gamma, tau))  # shape=(n_d, n_tau)

    Id = np.array(
        [mieScattInt(theta, di, params.wavelength, ri_particle_complex) for di in d]
    )
    G = Id*N / np.sum(Id*N)  # shape=(n_d,)
    g1 = np.einsum('i,ij->j', G, g1i)  # shape=(n_tau,)
    return g1

def mieScattInt(theta:float, d:float, wavelength:float, ri_particle_complex:complex) -> float:
    '''
    只能计算单个角度单个粒径的米氏散射强度
    '''
    m = ri_particle_complex
    x = np.pi*d/wavelength
    mu = np.cos(theta)
    S1, S2 = ps.MieS1S2(m, x, mu)
    return np.abs(S1)**2  # S1是复数，这里是模的平方的意思

##############################################################################


################### simulation related functions #############################

def genContinuousN(d:ndarray, mu:list, sigma:list, number:list):
    N = np.zeros_like(d)
    if not hasattr(mu, '__iter__'):
        mu, sigma, number = [mu], [sigma], [number]
    for m, sig, num in zip(mu, sigma, number):
        N += num * np.exp(-(d-m)**2/(2*sig**2))
    N = N/np.sum(N)
    return N

##############################################################################



class DlsData:
    def __init__(self, data:dict, load=False) -> None:
        params = data['params']  # Params object
        if isinstance(params, dict):
            self.params = Params(params)
        elif isinstance(params, Params):
            self.params = params
        self.angle = int(data['angle'])  
        self.theta = self.angle/180*np.pi  

        self.mode = data['mode']
        if 'sim_info' in data.keys():
            self.sim_info = data['sim_info']
        else:
            self.sim_info = None
        if 'exp_info' in data.keys():
            self.exp_info = data['exp_info']
        else:
            self.exp_info = None

        self.tau = np.array(data['tau']).flatten()
        self.g2 = np.array(data['g2']).flatten()
        self.baseline = data['baseline']
        self.intensity = np.sqrt(self.baseline)
        if load:
            self.beta = data['beta']
            self.g1square = np.array(data['g1square']).flatten()
            self.g1 = np.array(data['g1']).flatten()
        else:
            self.calcG1()

    def calcG1(self, pnum_fitbeta:int=10) -> None:
        # pnum_fitbeta是取前多少个点拟合beta的值
        Ctau = self.g2/self.baseline - 1
        self.beta = self._calcBeta(self.tau, Ctau, pnum_fitbeta)
        g1square = Ctau/self.beta
        self.g1square = g1square
        self.g1 = np.sign(g1square) * np.sqrt(np.abs(g1square))

    def _calcBeta(self, tau:ndarray, Ctau:ndarray, pnum_fitbeta:int) -> float:
        y = np.log(Ctau[:pnum_fitbeta])
        x = tau[:pnum_fitbeta]
        intercept = np.polyfit(x, y, 1)[-1]
        beta = np.exp(intercept)
        return beta

    def toDict(self) -> dict:
        dic = copy.deepcopy(self.__dict__)  # 深复制一个，否则更改__dict__会直接改变对象参数的值
        dic['tau'] = self.tau.tolist()
        dic['g2'] = self.g2.tolist()
        dic['g1'] = self.g1.tolist()
        dic['g1square'] = self.g1square.tolist()
        dic['params'] = self.params.toDict()
        return dic



class MdlsData:
    def __init__(self, dic:dict=None) -> None:
        '''dic is pre-saved dict by self.toDict()'''
        self.data = {}
        if isinstance(dic, dict):
            self.params = Params(dic['params'])
            self.mode = dic['mode']
            self.sim_info = dic['sim_info']
            self.exp_info = dic['exp_info']
            for dlsdata_dict in dic['data'].values():
                angle = dlsdata_dict['angle']
                self.data[angle] = DlsData(dlsdata_dict, load=True)
            self.sortDataDict()
            
    def addDlsData(self, dlsdata:DlsData):
        self.data[dlsdata.angle] = dlsdata
        self.updateParams(dlsdata)
        self.sortDataDict()

    def sortDataDict(self):
        '''保持data中角度顺序始终由小到大'''
        angles = list(self.data.keys())
        angles.sort()
        temp_dict = copy.deepcopy(self.data)
        self.data = {}
        for angle in angles:
            self.data[angle] = temp_dict[angle]

    def updateParams(self, dlsdata:DlsData):
        self.params = dlsdata.params
        self.mode = dlsdata.mode
        self.sim_info = dlsdata.sim_info
        self.exp_info = dlsdata.exp_info

    def toDict(self) -> dict:
        dic = copy.deepcopy(self.__dict__)  # 深复制一个，否则更改__dict__会直接改变对象参数的值
        dic['params'] = self.params.toDict()
        for angle in self.data.keys():
            dic['data'][angle] = self.data[angle].toDict()
        return dic


class DlsSimulator:
    default_sim_info = {
        'g2_error_scale': 0.0005,
        'intensity_at_smallest_angle': (600e3, 1000e3),  # 随机生成600k~1M cps的强度
        'intensity_error_scale': 0.001,
        'SLS_baseline': (0.1e3, 2e3),   # 随机生成0.1k~2k cps的SLS基线
        'beta': (0.2, 0.8)  # 随机生成0.2~0.8的beta
    }
    def __init__(self, d:ndarray, N:ndarray, Nd_mode:str, tau:ndarray, params:Params) -> None:
        '''
        Nd_mode = 'discrete' or 'continuous'
        '''
        self.Nd_mode = Nd_mode
        self.tau = np.array(tau).flatten()
        self.d = np.array(d).flatten()
        self.N = np.array(N).flatten()
        self.N = self.N / np.sum(self.N)  # 将N的分布归一化为和等于1
        self.params = params

        self.sim_info = copy.deepcopy(self.default_sim_info)
        t = self.sim_info['intensity_at_smallest_angle']
        self.sim_info['intensity_at_smallest_angle'] = t[0] + (t[1]-t[0])*np.random.rand()
        t = self.sim_info['SLS_baseline']
        self.sim_info['SLS_baseline'] = t[0] + (t[1]-t[0])*np.random.rand()
        t = self.sim_info['beta']
        self.sim_info['beta'] = t[0] + (t[1]-t[0])*np.random.rand()
        self.sim_info['d'] = self.d.tolist()  # 便于存储
        self.sim_info['N'] = self.N.tolist()  # 便于存储
        self.sim_info['Nd_mode'] = Nd_mode
        # 计算G便于画图，90度的
        ri_particle_complex = complex(self.params.ri_particle_real, self.params.ri_particle_img)
        I = np.array(
            [mieScattInt(np.pi/2, di, self.params.wavelength, ri_particle_complex) for di in self.d]
        )
        G = self.N*I / np.sum(self.N*I)
        self.sim_info['G'] = G.tolist()

    def simMdlsData(self, angles:list) -> MdlsData:
        I_list = []
        for angle in angles:
            theta = angle/180*np.pi
            ri_particle_complex = complex(self.params.ri_particle_real, self.params.ri_particle_img)
            Id = np.array(
                [mieScattInt(theta, di, self.params.wavelength, ri_particle_complex) for di in self.d]
            )
            I = np.sum(Id*self.N)
            I_list.append(I)
        min_angle = min(angles)
        index = angles.index(min_angle)
        scale = self.sim_info['intensity_at_smallest_angle'] / I_list[index]
        I_list = [scale*Ii for Ii in I_list]

        mdlsdata = MdlsData()
        for angle, intensity in zip(angles, I_list):
            dlsdata = self.simDlsData(angle, intensity)
            mdlsdata.addDlsData(dlsdata)
        return mdlsdata

    def simDlsData(self, angle:float, intensity:float) -> DlsData:
        '''
        模拟单个角度的DLS数据
        intensity是指用于模拟多角度DLS数据情况下该角度的静态散射强度，即<I>，
        一般来说g2在tau趋向于正无穷时等于<I>^2
        '''
        g1 = calcG1(self.tau, self.d, self.N, angle, self.params)
        beta = self.sim_info['beta']

        scale = self.sim_info['intensity_error_scale']
        baseline = intensity**2 + scale**2 * np.random.randn()*np.random.randn()

        beta = self.sim_info['beta']
        g2 = baseline * (1 + beta*g1**2)
        scale = self.sim_info['g2_error_scale']
        g2_error = scale * baseline * np.random.randn(g2.size)*np.random.randn(g2.size)
        g2 = g2 + g2_error

        data = {
            'params': self.params,
            'angle': angle,
            'mode': 'sim',
            'sim_info': self.sim_info,
            'tau': self.tau,
            'g2': g2,
            'baseline': baseline,
        }
        return DlsData(data)

    
def loadBrookhavenDatFile(filename:str, comments:str=None, baseline:str='calculated') -> DlsData:
    mode = 'exp'
    with open(filename, 'r', encoding='utf-8') as f:
        lines = list(f.readlines())
    lines = [line.rstrip('\n') for line in lines] # 删除末尾的换行符
    raw_data = '\n'.join(lines)
    params_dict = {
    'wavelength': float(lines[9]),           # nanometer
    'temperature': float(lines[10]),          # Kelvin
    'viscosity': float(lines[11]),           # cP
    'ri_liquid': float(lines[13]),
    'ri_particle_real': float(lines[14]),
    'ri_particle_img': float(lines[15])
    }
    params = Params(params_dict=params_dict)
    angle = int(float(lines[8]))
    try:
        baseline = float(baseline)  # in case of inputed number
    except:
        if 'mea' in baseline.lower(): 
            baseline = float(lines[22])
        else:  # default use calculated baseline
            baseline = float(lines[21])
    
    # 新旧软件输出不同
    if ',' in lines[37]:
        delimiter = ', ' 
    else:
        delimiter = ' '
    tau, g2 = [], []
    for line in lines[37:-4]:
        tau.append(float(line.split(delimiter)[0]))
        g2.append(float(line.split(delimiter)[1]))
    exp_info = {
        'filename': os.path.basename(filename),
        'comments': comments,
        'sample_id': lines[-4],
        'operator_id': lines[-3],
        'date': lines[-2],
        'time': lines[-1],
        'raw_data': raw_data
    }
    data = {
        'params': params,
        'angle': angle,
        'mode': mode,
        'exp_info': exp_info,
        'baseline': baseline,
        'tau': tau,
        'g2': g2
    }
    dlsdata = DlsData(data)
    return dlsdata


if __name__ == '__main__':
    from PlotUtils import plotSimData
    
    #d = [10, 500]
    #N = [1e11, 1]
    d = np.linspace(1, 500, num=50)
    N = genContinuousN(d, [10, 300], [3, 5], [10, 1])
    tau = np.logspace(0, 5, num=200)
    params = Params()
    sim = DlsSimulator(d, N, 'continuous', tau, params)
    mdlsdata = sim.simMdlsData(range(30, 151, 15))

    plotSimData(mdlsdata, filename='test.png')

    # save test
    if False:
        import json
        with open('test.json', 'w') as f:
            json.dump(mdlsdata.toDict(), f, indent=4)

        with open('test.json', 'r') as f:
            mdlsdata = MdlsData(json.load(f))

    # plot
    if False:
        import matplotlib.pyplot as plt
        plt.subplot(121)
        for dlsdata in mdlsdata.data.values():
            plt.plot(dlsdata.tau, dlsdata.g1**2)
        #plt.plot(mdls.data[90].tau, mdls.data[90].g1**2)
        plt.xscale('log')

        plt.subplot(122)
        l1, l2 = [], []
        for dlsdata in mdlsdata.data.values():
            l1.append(dlsdata.angle)
            l2.append(dlsdata.intensity)
        plt.plot(l1, l2)
        #plt.yscale('log')

        plt.show()

        print('done')