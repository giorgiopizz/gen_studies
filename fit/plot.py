import itertools
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
# import concurrent.futures

import matplotlib.pyplot as plt
import scipy.interpolate
import uproot
import mplhep as hep
from skimage import measure

hep.style.use(hep.style.CMS)  # For now ROOT defaults to CMS


def set_label():
    hep.cms.label("osWW", data=False, lumi=100.0, exp="")

ops = ['cWtil', 'cHWtil']
variables = ['mjj', 'detajj', 'dphijj']
ops = ['cWtil']
variables = ['mjj', 'detajj', 'dphijj', 'ptj1', 'events']

cmap_petroff = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]


def plot1d(filenames, op, labels=None, debug=False):
    for ifile, filename in enumerate(filenames):
        f = uproot.open(filename)
        tree = f["limit"]
        c1 = tree[op].array().to_numpy()
        res = tree["nll"].array().to_numpy()
        f.close()
        interp = scipy.interpolate.interp1d(c1, res)
        x = np.linspace(min(c1), max(c1), 10000)
        z = interp(x)
        # xmins = x[np.where(np.diff(np.sign(z - np.tile(1.0, z.shape))))]
        # print(z)
        mask = ~np.isnan(z)
        x = x[mask]
        z = z[mask]
        if len(z) == 0:
            continue

        set_label()

        color = cmap_petroff[ifile]
        label = filename
        if labels:
            label = labels[ifile]

        target = 1.0
        target = 3.84
        xmins = x[np.where(np.diff(np.sign(z - np.tile(target, z.shape))))]
        if len(xmins) == 0:
            print(op, filename)
            continue
        txt = f"[{round(xmins[0], 2)}:{round(xmins[1], 2)}]"
        label += " " + txt
        # kwargs = {'fontsize': 18}
        # plt.text(xmins[0], target, str(round(xmins[0], 2)), ha="left", color=color, **kwargs)
        # plt.text(xmins[1], target, str(round(xmins[1], 2)), ha="right", color=color, **kwargs)
        # target = 3.84
        # xmins = x[np.where(np.diff(np.sign(z - np.tile(target, z.shape))))]
        # plt.text(xmins[0], target, str(round(xmins[0], 2)), ha="left", color=color, **kwargs)
        # plt.text(xmins[1], target, str(round(xmins[1], 2)), ha="right", color=color, **kwargs)
        # print(xmins[:5])

        if debug:
            plt.plot(c1, res, "o", markersize=2, color=color)
        plt.plot(x, z, label=label, color=color)

    plt.plot(x, np.ones_like(x) * 1, color="red", linestyle="dashed")
    plt.plot(x, np.ones_like(x) * 3.84, color="green", linestyle="dashed")
    plt.ylabel("$-2\\Delta LL$", loc="top")
    plt.xlabel(op, loc="right")
    plt.ylim(0.0, None)
    plt.legend()
    plt.savefig(f"scan_1d_{op}.png")


def plot2d(filenames, ops, labels=None, debug=False):
    for ifile, filename in enumerate(filenames):
        f = uproot.open(filename)
        tree = f["limit"]

        c1 = tree[ops[0]].array().to_numpy()
        c2 = tree[ops[1]].array().to_numpy()
        res = tree["nll"].array().to_numpy()
        f.close()

        c1c2 = np.array([c1, c2]).T
        interp = scipy.interpolate.LinearNDInterpolator(c1c2, res)
        x = np.linspace(min(c1), max(c1), 100)
        y = np.linspace(min(c2), max(c2), 100)
        XY = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
        X = XY[:, 0]
        Y = XY[:, 1]
        # sys.exit()
        Z = interp(X, Y)

        set_label()

        if debug:
            im = plt.pcolormesh(
                x, y, Z.reshape(x.shape[0], y.shape[0]), vmin=0.0, vmax=15
            )
            plt.colorbar(im)

        contours = measure.find_contours(Z.reshape(x.shape[0], y.shape[0]), 2.0)
        color = "red"
        color = cmap_petroff[ifile]

        label = filename
        if labels:
            label = labels[ifile]

        for contour in contours:
            _x = ((contour[:, 1]) / (x.shape[0] - 1)) * (max(x) - min(x)) + min(x)
            _y = ((contour[:, 0]) / (y.shape[0] - 1)) * (max(y) - min(y)) + min(y)
            plt.plot(_x, _y, linewidth=2, color=color, label=label)

        contours = measure.find_contours(Z.reshape(x.shape[0], y.shape[0]), 5.99)
        # color = "orange"
        for contour in contours:
            _x = ((contour[:, 1]) / (x.shape[0] - 1)) * (max(x) - min(x)) + min(x)
            _y = ((contour[:, 0]) / (y.shape[0] - 1)) * (max(y) - min(y)) + min(y)
            plt.plot(_x, _y, linewidth=2, color=color, linestyle="dashed")
    plt.xlabel(ops[0], loc='right')
    plt.ylabel(ops[1], loc='top')
    plt.legend()
    plt.savefig(f"scan_2d_{ops[0]}_{ops[1]}.png")


# tree = uproot.open('results.root')['limit']
# c1 = tree['cWtil'].array().to_numpy()
# res = tree['nll'].array().to_numpy()
# plot1d(['results.root', 'results_1d_stat.root'])
# plot1d(['results.root'])

# tree = uproot.open('results.root')['limit']
# c1 = tree['cWtil'].array().to_numpy()
# c2 = tree['ctWIm'].array().to_numpy()
# res = tree['nll'].array().to_numpy()
# plot2d(c1, c2, res)
# plot1d(['results.root'], 'cHWtil', labels=['mjj'], debug=True)
# plot2d(["results.root"], ["cWtil", "cHWtil"], labels=["mjj"], debug=False)
tot_ops = [
    "ctBIm",
    "cHtbIm",
    "cbBIm",
    "cHWtil",
    "cQtQb8Im",
    "ctWIm",
    "cHGtil",
    "cHBtil",
    "cHWBtil",
    "cQtQb1Im",
    "cbWIm",
    "cWtil",
    "cbHIm",
]
# tot_ops = tot_ops[:6]

# variables = ["mjj"]
# for ops in list(itertools.combinations(tot_ops, 2)):
#     # for op in tot_ops:
#     ops = list(ops)
#     # ops = [op]
#     filenames = []
#     labels = []
#     for variable in variables:
#         filenames.append(f"results/{variable}_{'_'.join(ops)}.root")
#         labels.append(variable)

# job = [(['cWtil'], ['mjj'])]
# for ops, variables in job:
#     filenames = []

filenames = ['results/mjj_cWtil.root', 'test1.root']
ops = ['cWtil']
labels = None
# for filenames, ops, labels in job:
plt.clf()
if len(ops) == 1:
    plot1d(filenames, ops[0], labels=labels, debug=False)
else:
    plot2d(filenames, ops, labels=labels, debug=True)
# plot2d(["results/mjj_cWtil_cHWtil.root", "results/detajj_cWtil_cHWtil.root"], ["cWtil", "cHWtil"], labels=["mjj", "detajj"], debug=False)
# plot2d(["results.root"], ["cWtil", "cHGtil"], labels=["mjj"], debug=False)
# plot2d(['results.root'], ['cWtil', 'cHGtil'], debug=True)
