import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

def plot_bullseye(data,mu,vmin=None,rd=None, vmax=None, savepath=None,cmap='plasma', label='GPRS (%)', 
                  std=None,cbar=False,color='white', fs=14, xshift=0, yshift=0, ptype='mesh',frac=False):
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6,6))
    

    rho     = np.arange(0,4,4.0/data.shape[1])
    Theta   = np.deg2rad(range(data.shape[0]+1))
    # Theta   = np.deg2rad(range(360)) 
    [th, r] = np.meshgrid(Theta, rho)
    data = np.append(data,data[359:,],axis=0)

    vmin = data.min() - (data.max()-data.min())*1/5
    vmax = data.max() - (data.max()-data.min())*1/5
    
    if ptype == 'mesh':
        im = ax.pcolormesh(r*np.cos(Theta), r*np.sin(Theta), data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    else:
        im = ax.contourf(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    
    
    if cbar:
        cbar = plt.colorbar(im, cax=fig.add_axes([0.05, 0.05, 1/3, 0.03]), orientation='horizontal')
        # cbar = plt.colorbar(im, orientation='horizontal')
        new_ticks = []
        new_ticks_labels = []
        for i,tick in enumerate(cbar.ax.get_xticks()):
            if i % 2 == 0:
                new_ticks.append(np.round(tick))
                new_ticks_labels.append(str(int(np.round(tick))))

        cbar.set_ticks(new_ticks)
        cbar.set_ticklabels(new_ticks_labels)

        if vmin is not None:
            cbar.set_ticks([vmin, (vmax+vmin)/2.0, vmax])
            cbar.set_ticklabels(['%d'%(i) for i in [vmin, (vmax+vmin)/2.0, vmax]])
        
        cbar.set_ticks([2,2.5])
        cbar.set_ticklabels(['0%','100%'])

        cbar.ax.tick_params(labelsize=10)
        # cbar.set_label(label, fontsize=26, weight='bold')

    ax.axis('off')

    draw_circle(ax, np.array(mu), color=color, fs=fs, xshift=xshift, yshift=yshift)

    if savepath is not None:
        if not cbar:
            plt.tight_layout()
        plt.savefig(savepath, dpi=600)

    # plt.tight_layout()
    # plt.show()
    
def draw_circle(ax, mu, width=1, fs=2, xshift=0, yshift=0, color='white'):

    
    circle1 = plt.Circle((0,0), 1, color='white', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='white', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='white', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='white', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)

    plt.xlim(-4.01, 4.01)
    plt.ylim(-4.01, 4.01)
    
    j = 1
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='white', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(3.5, theta_i - 2*np.pi/12)
        # ax.text(xi-.3-xshift, yi-yshift-.1, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color)
        xi, yi = polar2cart(2.5, theta_i - 2*np.pi/12)
        # ax.text(xi-.3-xshift, yi-yshift-.1, '%.2f' %(mu[j+6]), weight='bold', fontsize=fs, color=color); j += 1
        
    j += 6
    LABELS = ['ANT', 'SEPT', 'INF', 'LAT']
    for i in range(4):
        theta_i = i*90*np.pi/180  + 45*np.pi/180
        xi, yi = polar2cart(1, theta_i)
        xf, yf = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='white', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(1.5, theta_i - 2*np.pi/8)

        # ax.text(xi-.3-xshift, yi-yshift-.1, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color); j += 1
        xi, yi = polar2cart(5, theta_i - 2*np.pi/8)

    # ax.text(-.3-xshift, 0-yshift-.1, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color)
    
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
