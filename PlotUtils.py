from math import log
import matplotlib.pyplot as plt
from cycler import cycler
import visdom
import torch
import numpy as np

class MdlsNNLogger:
    def __init__(self, epoch_num, mdls_tensordata, model, diameter_distribution_xtype='log', environment='MdlsNN'):
        self.epoch_num = epoch_num
        self.viz = visdom.Visdom(env=environment)
        self.mdls_tensordata = mdls_tensordata
        self.model = model
        self.diameter_distribution_xtype = diameter_distribution_xtype
        self.windows = {
            'loss': 'loss',
            'g1square': 'g1square',
            'distribution': 'distribution'
        }
        # 初始化loss函数这样相当于清空
        self.viz.line(
            X=[0],
            Y=[0],
            win=self.windows['loss'],
            name='loss',
            opts={'ytype':'log', 'title':'Loss'}
            )
        self.initG1square()
        self.updateNG()

        
    def log(self, epoch_index, loss, update_g1_node=10000):
        '''
        loss: float
        model: MdlsModel object
        '''
        self.viz.line(
            X=[epoch_index],
            Y=[loss],
            win=self.windows['loss'],
            update='append',
            opts={'ytype':'log', 'title':'Loss'}
        )
        self.updateNG()
        if epoch_index % update_g1_node == 0:
            self.updateG1square()

    def initG1square(self, exp_data_plot_type='line'):
        '''legend of each exp data is angle + exp'''
        # plot original exp data as scatter
        angle_list = list(self.mdls_tensordata.data.keys())
        if exp_data_plot_type == 'scatter':
            # scatter plot of exp data
            data = self.mdls_tensordata.data[angle_list[0]]
            points = torch.stack((data.tau, data.g1square)).T
            points_y = torch.ones_like(data.tau, dtype=torch.int).unsqueeze(1)
            name = str(data.angle.item()) + 'exp'
            self.viz.scatter(
                X=points,
                win=self.windows['g1square'],
                name=name,
                update=None,
                opts={'xtype':'log', 'title':'g1square', 'markersize':4}
            )
            for i in range(1, len(angle_list)):
                data = self.mdls_tensordata.data[angle_list[i]]
                points = torch.stack((data.tau, data.g1square)).T
                name = str(data.angle.item()) + 'exp'
                self.viz.scatter(
                    X=points,
                    win=self.windows['g1square'],
                    name=name,
                    update='append',
                    opts={'markersize':4}
                )
        else:
            # line plot of exp data
            tau = self.mdls_tensordata.data[angle_list[0]].tau
            g1square_list, names = [], []
            for angle in self.mdls_tensordata.data.keys():
                data = self.mdls_tensordata.data[angle]
                g1square_list.append(data.g1square)
                names.append(str(angle)+'exp')
            g1square = torch.stack(g1square_list).T
            self.viz.line(
                X=tau,
                Y=g1square,
                win=self.windows['g1square'],
                update=None,
                opts={'xtype':'log', 'title':'g1square'}
            )

        # plot fit data as line
        tau = self.model.tau
        g1square = self.model.getG1square(to_numpy=False)
        for i in range(len(g1square)):
            g1square_i = g1square[i]
            name = str(self.model.angle[i].item()) + 'fit'
            self.viz.line(
                X=tau,
                Y=g1square_i,
                name=name,
                win=self.windows['g1square'],
                update='append',
                opts={'linecolor':np.array([[0,0,0]])}
            )

    def updateG1square(self):
        # plot fit data as line
        tau = self.model.tau
        g1square = self.model.getG1square(to_numpy=False)
        for i in range(len(g1square)):
            g1square_i = g1square[i]
            name = str(self.model.angle[i].item())+'fit'
            self.viz.line(
                X=tau,
                Y=g1square_i,
                name=name,
                win=self.windows['g1square'],
                update='replace',
                opts={'linecolor':np.array([[0,0,0]])}
            )

    def updateNG(self):
        #theta1 = self.model.all_theta[0]
        #theta2 = self.model.all_theta[-1]
        d = self.model.getD(to_numpy=False)
        N = self.model.getN(to_numpy=False)
        G = self.model.getG(to_numpy=False)
        #d, G2 = self.model.getIntensityDistribution(theta=theta2, to_numpy=False)
        y = torch.stack((N, G)).T
        self.viz.line(
            X=d,
            Y=y,
            win=self.windows['distribution'],
            update=None,
            opts={
                'xtype':self.diameter_distribution_xtype,
                'title':'Diameter Distribution',
                'legend':['number', 'intensity@90°']
            }
        )
        

def plotAxisLoss(ax, iter, loss, *args, **kwargs):
    ax.plot(iter, loss, *args, **kwargs)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Loss')
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')

def plotAxisDiameterDistribution(ax, d, distribution, log_d=True, *args, **kwargs):
    ax.plot(d, distribution, *args, **kwargs)
    if log_d:
        ax.set_xscale('log')
    ax.set_title('Diameter Distribution')
    ax.set_xlabel('diameter (nm)')

def plotAxisCorrelationFunction(ax, tau_list, g_list, *args, **kwargs):
    for i in range(len(g_list)):
        ax.plot(tau_list[i], g_list[i], *args, **kwargs)
    ax.set_xscale('log')
    ax.set_title('Correlation Function')
    ax.set_xlabel('$\\rm \\tau\;(\mu s)$')
    ax.set_ylabel('$\\rm |g_1|^2$')

def plotSimData(mdlsdata, show=False, filename=None):
    fig = plt.figure(figsize=(12, 8), dpi=300, facecolor='white')
    ax1 = plt.subplot2grid((2,3), (0,0), colspan=2)
    plotAxisCorrelationFunction(
        ax1,
        [dlsdata.tau for dlsdata in mdlsdata.data.values()],
        [dlsdata.g1square for dlsdata in mdlsdata.data.values()]
    )

    ax2 = plt.subplot2grid((2,3), (0,2))
    ax2.plot(
        list(mdlsdata.data.keys()),
        [dlsdata.intensity for dlsdata in mdlsdata.data.values()]
    )
    ax2.set_title('Scattering Intensity')
    ax2.set_xlabel('angle (degree)')
    ax2.set_ylabel('intensity (cps)')

    ax3 = plt.subplot2grid((2,3), (1,0), colspan=3)
    d = np.array(mdlsdata.sim_info['d'])
    N = np.array(mdlsdata.sim_info['N'])
    width = 0.1 * d  # automatically determine the width of bar according to data
    if mdlsdata.sim_info['Nd_mode'] == 'discrete':
        ax3.bar(d, N, width, color='lightskyblue', label='simulated number distribution')
    elif mdlsdata.sim_info['Nd_mode'] == 'continuous':
        ax3.fill(d, N, color='lightskyblue', edgecolor=None, alpha=0.5, label='simulated number distribution')

    fig.tight_layout()
    if show:
        plt.show()
    if filename != None:
        fig.savefig(filename)













def plotFitExperimentResult(fit_exp, show=False, figname=None, figsize=(15,15), facecolor='white', **kwargs):
    # kwargs pass to plt.figure()
    fig = plt.figure(figsize=figsize, facecolor=facecolor, **kwargs)
    ax1 = plt.subplot2grid((3,3), (0,0))
    plotAxisLoss(ax1, fit_exp.record_epoch, fit_exp.record_loss, color='gray')
    
    ax2 = plt.subplot2grid((3,3), (0,1), colspan=2)
    ##### 自定义cycler #####
    # 以 Set3 colormap 为例
    custom_cycler = cycler("color", plt.cm.Set3.colors)
    ax2.set_prop_cycle(custom_cycler)
    #######################
    plotAxisCorrelationFunction(
        ax2,
        [data.tau for data in fit_exp.dls_data_set.dls_data_list],
        [data.g1square for data in fit_exp.dls_data_set.dls_data_list],
        '.'
    )
    plotAxisCorrelationFunction(
        ax2,
        [data.tau for data in fit_exp.dls_data_set.dls_data_list],
        fit_exp.result_g1square,
        'k-'
    )

    # diameter distribution with no log_d
    ax3 = plt.subplot2grid((3,3), (1,0), colspan=3)
    if fit_exp.mode == 'sim':
        N = fit_exp.sim_N/np.sum(fit_exp.sim_N)
        N = N / np.max(N) * np.max(fit_exp.result_N)
        G = fit_exp.sim_G/np.sum(fit_exp.sim_G)
        G = G / np.max(G) * np.max(fit_exp.result_G)
        width = (fit_exp.result_d.max() - fit_exp.result_d.min())/100  # automatically determine the width of bar according to data
        if len(fit_exp.sim_d) <= 10:
            ax3.bar(fit_exp.sim_d, N, width, color='lightskyblue', label='simulated number distribution')
            ax3.bar(fit_exp.sim_d, G, width, color='violet', label='simulated intensity distribution@small angle')
        else:
            ax3.fill(fit_exp.sim_d, N, color='lightskyblue', edgecolor=None, alpha=0.5, label='simulated number distribution')
            ax3.fill(fit_exp.sim_d, G, color='violet', edgecolor=None, alpha=0.5, label='simulated intensity distribution@small angle')
    plotAxisDiameterDistribution(ax3, fit_exp.result_d, fit_exp.result_N, log_d=False, color='blue', label='number weighted')
    plotAxisDiameterDistribution(ax3, fit_exp.result_d, fit_exp.result_G, log_d=False, color='fuchsia', label='intensity weighted@small angle')
    ax3.legend()

    # diameter distribution with log_d
    ax4 = plt.subplot2grid((3,3), (2,0), colspan=3)
    if fit_exp.mode == 'sim':
        N = fit_exp.sim_N/np.sum(fit_exp.sim_N)
        N = N / np.max(N) * np.max(fit_exp.result_N)
        G = fit_exp.sim_G/np.sum(fit_exp.sim_G)
        G = G / np.max(G) * np.max(fit_exp.result_G)
        width = 0.1 * fit_exp.sim_d  # automatically determine the width of bar according to data
        if len(fit_exp.sim_d) <= 10:
            ax4.bar(fit_exp.sim_d, N, width, color='lightskyblue', label='simulated number distribution')
            ax4.bar(fit_exp.sim_d, G, width, color='violet', label='simulated intensity distribution@small angle')
        else:
            ax4.fill(fit_exp.sim_d, N, color='lightskyblue', edgecolor=None, alpha=0.5, label='simulated number distribution')
            ax4.fill(fit_exp.sim_d, G, color='violet', edgecolor=None, alpha=0.5, label='simulated intensity distribution@small angle')
    plotAxisDiameterDistribution(ax4, fit_exp.result_d, fit_exp.result_N, log_d=True, color='blue', label='number weighted')
    plotAxisDiameterDistribution(ax4, fit_exp.result_d, fit_exp.result_G, log_d=True, color='fuchsia', label='intensity weighted@small angle')
    ax4.legend()

    fig.set_tight_layout(True)

    if show:
        plt.show()
    if figname:
        fig.savefig(figname)


    


if __name__ == '__main__':
    import numpy as np
    viz = visdom.Visdom()
    x1 = np.linspace(-10, 10)
    y1 = np.sin(x1)
    y2 = np.sin(x1-2)

    '''
    visdom里同一个窗口内画多条线，必须在第一句里面先把所有要画的线都画出来，然后后面再更新。
    一次画多条线，输入的X与Y应该具有多个column
    第一次应给定legend，后面使用name参数指定更新哪条线
    '''

    # line plot
    viz.line(X=np.hstack([x1,x1]).T, Y=np.hstack([y1,y2]).T, win='line test', opts=dict(legend=['l1', 'l2']))
    viz.line(X=x1, Y=y1+1, win='line test', name='l1', update='replace')
    viz.line(X=x1, Y=y2+2, win='line test', name='l2', update='replace')
    
    # scatter plot
    '''
    visdom里画散点图也是很反人类。。。
    总之散点的坐标都应该放在X里，shape=(n,2)或者(n,3)如果要画三维散点
    至于不同的系列，则通过y指定，每一个点对应的y的整数值为一个系列
    '''
    pts1 = np.stack((x1, y1)).T
    pts2 = np.stack((x1, y2)).T
    pts = np.vstack([pts1, pts2])
    pts_y = np.hstack([1*np.ones_like(x1, dtype='int'), 2*np.ones_like(x1, dtype='int')]).T
    viz.scatter(X=pts, Y=pts_y, win='scatter test', opts=dict(legend=['s1', 's2']))


    '''下面是一个例子，如何在一个图中添加其他的线或者散点，以及怎么在同一个图中同时画线和散点
    总之就是第一个数据画的时候没有update参数，后面的用update='appennd'添加，但是每一个数据都要指定name '''
    viz.scatter(
        X = [[1,1], [2,2], [3,3],[4,4]],
        win='line',
        name='scatter',
        update=None
    )
    viz.line(
        X=[1, 2, 3, 4],
        Y=[1, 4, 9, 16],
        win="line",
        name='line1',
        update='append'
    )
    viz.line(
        X=[1, 2, 3, 4],
        Y=[0.5, 2, 4.5, 8],
        win="line",
        name='line2',
        update='append'
    )