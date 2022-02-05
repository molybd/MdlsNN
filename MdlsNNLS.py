import numpy as np
import scipy.optimize
from torch.functional import norm
from DataUtils import calcMieScatteringIntensity, genD

def mdlsNNLS(dls_data_set, d):
    '''
    conduct NNLS algorithm on multiangle DLS data simultaneously
    '''
    kb = 1.38064852e-23
    tau = dls_data_set.tau  # shape == (n_tau,)
    wavelength = dls_data_set.wavelength
    temperature = dls_data_set.temperature
    viscosity = dls_data_set.viscosity
    RI_liquid = dls_data_set.RI_liquid
    RI_particle_complex = dls_data_set.RI_particle_complex
    theta_list = [data.theta for data in dls_data_set.dls_data_list]

    Diffuse = (kb*temperature) / (3*np.pi*viscosity*d)  # shape == (n_d,)
    A_list = []
    for theta in theta_list:
        q = (4*np.pi*RI_liquid*np.sin(theta/2)) / (wavelength)  # float
        Gamma = Diffuse * q**2 * 1e24  # make unit μm^-1   # shape == (n_d,)
        exp = -np.einsum('i,j->ij', Gamma, tau)  # shape == (n_d, n_tau)
        exp = np.exp(exp)  # shape == (n_d, n_tau)
        I_d = [calcMieScatteringIntensity(theta, di, wavelength, RI_particle_complex) for di in d]
        I_d = np.array(I_d)  # shape == (n_d,)
        A_theta = np.einsum('i,ij->ij', I_d, exp)  # shape == (n_d, n_tau)
        A_theta = A_theta.T  # shape == (n_tau, n_d)
        A_list.append(A_theta)
    A = np.vstack(A_list)  # shape == (n_theta*n_tau, n_d)

    l = [data.g1 for data in dls_data_set.dls_data_list]
    b = np.hstack(l)  # shape == (n_theta*n_tau,)

    N, norm = scipy.optimize.nnls(A, b)
    return N, norm



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from DataUtils import calcIntensityDistribution, DlsData, DlsDataSet, simulateDlsData

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

    d, N = [10, 300], [2e10, 5000]  # 大粒子为主的体系 （使用g1效果不太好，即使loss还挺小的）
    #d, N = [10, 300], [2e10, 1]  # 小粒子为主的体系 （使用g1效果还不错）
    #d, N = [8, 70, 300], [1e13, 1e8, 1e3]  # 中等离子为主的三峰体系 （使用g1效果还不错）
    #d, N = [8, 70, 300], [1e12, 1e8, 1e3]  # 中等离子为主的三峰体系 （使用g1效果一般，即使loss还挺小的）
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
        plt.plot(dls_data1.tau, dls_data1.g1)
        dls_data_set.addDlsData(dls_data1)
    plt.xscale('log')
    plt.show()
    plt.savefig('test.png')

    d = genD(d_min=0.1, d_max=2000, d_num=50, log_d=True)
    N, norm = mdlsNNLS(dls_data_set, d)
    plt.plot(d, N)
    plt.savefig('test1.png')