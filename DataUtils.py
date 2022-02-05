import numpy as np
import PyMieScatt as ps

'''
====== Attention ======
at all place, the units are the regularly used units, c.a.
#param          #unit
tau          microsecond
angle        degree
wavelength   nanometer
viscosity    cP
temperature  Kelvin
diameter     nanometer
kb           J/K
at these units, the calculated Gamma should be multiplied by 1e24 to
become μm^(-1) unit that fits tau. for example, if gamma calculated is
2, then you should make it 2e24, then calculate g1 = exp(-gamma*tau)
'''

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
    'RI_particle_img': 0
}

class DlsDataSet:
    def __init__(self, mode='exp', dls_data_list=None):
        '''
        mode = 'exp' or 'sim'
        '''
        self.mode = mode
        self.angle_list = []
        self.theta_list = []
        if dls_data_list == None:
            self.dls_data_list = []
        else:
            self.dls_data_list = dls_data_list
            if len(self.dls_data_list) > 0:
                self.updateAttributes()

    def addDlsData(self, dls_data):
        self.dls_data_list.append(dls_data)
        self.updateAttributes()

    def updateAttributes(self):
        '''Assume that all the dls data in this data set share the
        same parameters, e.g. tau, wavelength, temperature etc. except
        for angles, g1, g2 and intensity
        '''
        dls_data = self.dls_data_list[-1]
        self.tau = dls_data.tau
        self.wavelength = dls_data.param_dict['wavelength']
        self.temperature = dls_data.param_dict['temperature']
        self.viscosity = dls_data.param_dict['viscosity']
        self.RI_liquid = dls_data.param_dict['RI_liquid']
        self.RI_particle_complex = complex(dls_data.param_dict['RI_particle_real'], dls_data.param_dict['RI_particle_img'])
        self.angle_list = [dls_data.angle for dls_data in self.dls_data_list]
        self.theta_list = [dls_data.theta for dls_data in self.dls_data_list]

class DlsData:
    def __init__(self, tau, g2, param_dict, mode='exp', sim_info_dict=None):
        '''
        mode = 'exp' or 'sim'
        '''
        self.tau = tau
        self.g2 = g2
        self.param_dict = param_dict
        self.angle = param_dict['angle']
        self.theta = self.angle/180*np.pi
        self.mode = mode
        if mode == 'exp':
            self.sim_info_dict = None
        elif mode == 'sim':
            self.sim_info_dict = sim_info_dict

        baseline = param_dict['baseline']
        self.intensity = np.sqrt(baseline)
        g1, g1square, beta = self.calcG1(tau, g2, baseline)
        self.g1 = g1
        self.g1square = g1square
        self.beta = beta

    def calcG1(self, tau, g2, baseline):
        B = baseline
        Ctau = g2/B - 1
        beta = self.calcBeta(tau, Ctau)
        g1square = Ctau/beta
        g1 = np.sign(g1square) * np.sqrt(np.abs(g1square))
        return g1, g1square, beta

    def calcBeta(self, tau, Ctau, point_num=10):
        y = np.log(Ctau[:point_num])
        x = tau[:point_num]
        intercept = np.polyfit(x, y, 1)[-1]
        beta = np.exp(intercept)
        return beta


def genD(d_min=1, d_max=1e5, d_num=50, log_d=True):
    if log_d:
        d = np.logspace(np.log10(d_min), np.log10(d_max), num=d_num)
    else:
        d  =np.linspace(d_min, d_max, num=d_num)
    return d


def simulateDlsData(d, N, param_dict, g2_error_scale=0.0005, baseline_scale=265166514e9, baseline_error_scale=0.0001, beta=0.43, tau=None):
    d, N = np.array(d), np.array(N)      # d in nanometer
    tau_min = param_dict['tau_min']                # microsec
    tau_max = param_dict['tau_max']              # microsec
    tau_num = param_dict['tau_num']            # microsec
    angle = param_dict['angle']                # degree
    wavelength = param_dict['wavelength']          # nanometer
    temperature = param_dict['temperature']          # Kelvin
    viscosity = param_dict['viscosity']           # cP
    RI_liquid = param_dict['RI_liquid']
    RI_particle_real = param_dict['RI_particle_real']
    RI_particle_img = param_dict['RI_particle_img']

    if tau:
        tau = np.array(tau)
    else:
        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), num=tau_num)
    
    theta = angle/180*np.pi
    beta = beta

    g1, I_sum = simulateG1(d, N, tau, theta, wavelength, temperature, viscosity, RI_liquid, complex(RI_particle_real, RI_particle_img))
    g1_without_error = g1

    # 因为未来处理的时候可能会需要用到强度数据，所以baseline也不能随便选取的，需要根据实际强度计算
    baseline_without_error = baseline_scale * I_sum**2  # 这里实际上是模拟了实际情况，即 g2(inf)=<I>**2
    error_on_baseline = baseline_error_scale * baseline_without_error * np.random.randn()*np.random.randn()
    baseline = baseline_without_error + error_on_baseline

    g2_without_error = baseline * (1 + beta*g1**2)
    error_on_g2 = g2_error_scale * baseline * np.random.randn(g1.size)*np.random.randn(g1.size)
    g2 = g2_without_error + error_on_g2

    sim_info_dict = {
        'd': d,
        'N': N,
        'g2_error_scale': g2_error_scale,
        'intensity': I_sum,
        'beta': beta,
        'error_on_g2': error_on_g2,
        'g1_without_error': g1_without_error,
        'g2_without_error': g2_without_error        
    }
    param_dict['baseline'] = baseline

    dls_data = DlsData(tau, g2, param_dict, mode='sim', sim_info_dict=sim_info_dict)
    return dls_data


def simulateG1(d, N, tau, theta, wavelength, temperature, viscosity, RI_liquid, RI_particle_complex):
    kb = 1.38064852e-23
    n = RI_liquid
    q = (4*np.pi*n*np.sin(theta/2)) / (wavelength)
    Diffuse = (kb*temperature) / (3*np.pi*viscosity*d)             # shape == (n_d,)
    Gamma = Diffuse * q**2 * 1e24  # make unit μm^-1               # shape == (n_d,)
    I_d = [calcMieScatteringIntensity(theta, di, wavelength, RI_particle_complex) for di in d]
    I_d = np.array(I_d)                                  # shape == (n_d,)
    I_sum = np.sum(I_d*N)                                       
    G = (I_d*N) / I_sum                              # shape == (n_d,)
    exp = np.exp(-1*np.einsum('i,j->ij', Gamma, tau))  # shape == (n_d, n_tau)
    g1 = np.einsum('i,ij->ij', G, exp)                   # shape == (n_d, n_tau)
    g1 = np.sum(g1, axis=0)
    return g1, I_sum


def genDiameterNumDistribution(mu, sigma, number, d_min=1, d_max=1e4, d_num=50, log_d=True, d_array=None):
    if d_array != None:
        d = np.array(d_array)
    else:
        if log_d:
            d = np.logspace(np.log10(d_min), np.log10(d_max), num=d_num)
        else:
            d = np.linspace(d_min, d_max, num=d_num)
    N = np.zeros_like(d)
    if not hasattr(mu, '__iter__'):
        mu, sigma, number = [mu], [sigma], [number]
    for m, sig, num in zip(mu, sigma, number):
        N += num * np.exp(-(d-m)**2/(2*sig**2))
    N = N / np.sum(N)
    return d, N


def calcMieScatteringIntensity(theta, d, wavelength, RI_particle_complex):
    '''
    please refer to:
    https://omlc.org/classroom/ece532/class3/mie_math.html
    For regular DLS aparatus, incident light polarization is perpendicular 
    to the scattering plane. And there is no analyzer for scattered light.
    Then, according to the aforementioned webpage
    I_s = I_s_parallel + I_s_perpendicular
        = const. * (S2^2*I_i_parallel + S1^2*I_i_perpendicular)
        = const. * S1^2*I_i_perpendicular

    Param unit:
    angle: radians
    d and wavelength must have same unit

    经过确认，这里输出的强度确实是与单个粒子散射总强度成正比
    即，确实符合瑞利散射中散射强度与粒径6次方成正比
    '''
    m = RI_particle_complex
    x = np.pi*d/wavelength
    mu = np.cos(theta)
    S1, S2 = ps.MieS1S2(m, x, mu)
    return np.abs(S1)**2


def calcIntensityDistribution(d, N, wavelength, RI_particle_complex, theta=np.pi/2):
    d, N = np.array(d), np.array(N)
    I_d = [calcMieScatteringIntensity(theta, di, wavelength, RI_particle_complex) for di in d]
    G = (I_d*N) / np.sum(I_d*N)
    return d, G


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pass