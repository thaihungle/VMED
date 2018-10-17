import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE



def sample_gaussian(dist, ns=1000):
    m, ls = np.split(dist, 2, axis=-1)
    s = np.exp(ls / 2)
    s = np.diag(s)
    return np.random.multivariate_normal(m, s,ns)

def sample_mgaussian(dists, mixw, ns=3000):
    num_mode = len(mixw)
    all_points = []
    for n in range(ns):
        di=np.random.choice(num_mode, 1, p=mixw)[0]
        p = sample_gaussian(dists[di], ns=1)[0]
        all_points.append(p)

    return all_points


def plot_tsne(dist1, mixw, dist2):
    point1s = sample_mgaussian(dist1, mixw, ns=100)
    point2s = sample_gaussian(dist2, ns=100)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=500, random_state=23)
    new_value1s = tsne_model.fit_transform(point1s)
    new_value2s = tsne_model.fit_transform(point2s)
    x1 = []
    y1 = []
    for value in new_value1s:
        x1.append(value[0])
        y1.append(value[1])

    x2 = []
    y2 = []

    for value in new_value2s:
        x2.append(value[0])
        y2.append(value[1])


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
    maxx = 0
    maxy = 0
    for i in range(len(x1)):
        ax[0].scatter(x1[i], y1[i], c='b')
        maxx = max(maxx, abs(x1[i]))
        maxy = max(maxy, abs(y1[i]))

    ax[0].set_xlim(-maxx*2, maxx*2)
    ax[0].set_ylim(-maxy*2, maxy*2)

    maxx = 0
    maxy = 0
    for i in range(len(x2)):
        ax[1].scatter(x2[i], y2[i], c='r')
        maxx = max(maxx, abs(x2[i]))
        maxy = max(maxy, abs(y2[i]))

    ax[1].set_xlim(-maxx * 2, maxx * 2)
    ax[1].set_ylim(-maxy * 2, maxy * 2)


    plt.show()



def plot_mgauss(dist1, mixw, dist2, xlim = (-5, 5), ylim = (-5, 5), xres = 500, yres = 500):

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x, y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    zz = None
    # print(dist1.shape)
    for di, d in enumerate(dist1):

        m, ls = np.split(d, 2, axis=-1)
        m = m[-2:]
        ls = ls[-2:]
        s = np.exp(ls / 2)

        s = np.diag(s)
        k = multivariate_normal(mean=m, cov=s)

        # print(m)
        # print(s)
        # print(mixw[di])
        # print('--')

        if di==0:
            zz = mixw[di]*k.pdf(xxyy)
        else:
            zz += mixw[di]*k.pdf(xxyy)

    # reshape and plot image
    img = zz.reshape((xres,yres))
    m, ls = np.split(dist2, 2, axis=-1)
    m = m[-2:]
    ls = ls[-2:]
    s = np.exp(ls / 2)

    s = np.diag(s)
    k = multivariate_normal(mean=m, cov=s)
    zz2 =  k.pdf(xxyy)
    img2 = zz2.reshape((xres, yres))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
    ax[0].imshow(img, cmap='Reds', extent=[-5, 5, -5, 5])
    ax[1].imshow(img2, cmap='Reds', extent=[-5, 5, -5, 5])

    plt.show()


if __name__ == '__main__':
    plot_mgauss()