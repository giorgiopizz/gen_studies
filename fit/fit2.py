import itertools
import json
import sys
import time

import iminuit
import matplotlib as mpl
import numpy as np
import pyhf

mpl.use("Agg")
import concurrent.futures

import matplotlib.pyplot as plt
import uproot
import scipy.interpolate

variable = "detajj"
variable = "dphijj"
# variable = "mjj"
variable = "ptj1"
variable = "events"


def read_ws(filename, ops):
    d = {}
    # with open(f"cards/{variable}.json") as serialized:
    with open(filename) as serialized:
        spec = json.load(serialized)

    workspace = pyhf.Workspace(spec)

    print(workspace)

    model = workspace.model()
    data = workspace.data(model)
    # pyhf.set_backend("numpy", "minuit")
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(tolerance=1e-3))

    # params = model.config.par_order
    params = model.config.par_names
    print(params)
    init_pars = [0.0 for _ in range(len(params))]
    par_bounds = [(-2000, 2000) for _ in range(len(params))]
    fixed_params = [False for _ in range(len(params))]

    init_pars[params.index("mu")] = 1.0
    fixed_params[params.index("mu")] = True
    par_bounds[params.index("mu")] = (0.8, 1.2)

    par_bounds[params.index("lumi")] = (-10, 10)

    for iparam, param in enumerate(params):
        if "stat" in param.lower():
            par_bounds[iparam] = (1e-10, 10.0)
            init_pars[iparam] = 1.0
            # fixed_params[iparam] = True
        elif param not in ["mu", "lumi"] + ops:
            fixed_params[iparam] = True

    _, nll_bestfit = pyhf.infer.mle.fit(
        data,
        model,
        init_pars=init_pars,
        par_bounds=par_bounds,
        fixed_params=fixed_params,
        return_fitted_val=True,
    )  # , return_result_obj=True)
    print(nll_bestfit)

    d["params"] = params
    d["data"] = data
    d["model"] = model
    d["init_pars"] = init_pars
    d["fixed_params"] = fixed_params
    d["par_bounds"] = par_bounds
    d["nll_bestfit"] = nll_bestfit
    return d


# print(pars_bestfit)
# print(objective)
# op1 = 'cWtil'
# coeff = -2.0
# init_pars[params.index(op1)] = coeff
# fixed_params[params.index(op1)] = True
# pars_bestfit, nll_bestfit2 = pyhf.infer.mle.fit(data, model, init_pars=init_pars, par_bounds=par_bounds, fixed_params=fixed_params, return_fitted_val=True)
# print(nll_bestfit2-nll_bestfit)
# sys.exit()


def fit1d(
    op1,
    coeff,
    params,
    init_pars,
    **kwargs,
):
    init_pars[params.index(op1)] = coeff
    # res.append(pyhf.infer.mle.twice_nll(init_pars, data, model))
    pdf = kwargs.pop("model")
    data = kwargs.pop("data")
    _fit = pyhf.infer.mle.fit(
        data,
        pdf,
        init_pars=init_pars,
        return_fitted_val=True,
        **kwargs,
    )
    # print(_fit)
    return _fit[1]

def scan(ops, limits, npoints, **kwargs):
    start = time.perf_counter_ns()
    params = kwargs.pop("params")
    init_pars = kwargs.pop("init_pars")
    fixed_params = kwargs.pop("fixed_params")
    par_bounds = kwargs.pop("par_bounds")
    nll_bestfit = kwargs.pop("nll_bestfit")

    for op in ops:
        fixed_params[params.index(op)] = True
    pdf = kwargs.pop('model')
    data = kwargs.pop('data')
    def func(pars):
        return -2 * pdf.logpdf(pars, data)
    m = iminuit.Minuit(func, init_pars, name=params)
    m.fixed = fixed_params
    m.limits = par_bounds
    # m.migrad(10000)
    # m.hesse()
    # print(m)
    # sys.exit()
    if len(ops) == 1:
        scan_values = np.linspace(-limits[0], limits[0], npoints).reshape(-1, 1)
    else:
        c1 = np.linspace(-limits[0], limits[0], int(np.sqrt(npoints)))
        c2 = np.linspace(-limits[1], limits[1], int(np.sqrt(npoints)))
        scan_values = np.dstack(np.meshgrid(c1, c2)).reshape(-1, 2)
    res = []
    for scan_value in scan_values:
        for iop, op in enumerate(ops):
            init_pars[params.index(op)] = scan_value[iop]
        # print(init_pars)
        for par_name in params:
            m.values[par_name] = init_pars[params.index(par_name)]
        m.migrad()
        m.hesse()
        res.append(m.fval)
        for par_name in good_ops:
            print(par_name, m.values[par_name])
        print(res[-1])
        # break
    # print(m)
    res = np.array(res)
    res -= nll_bestfit
    res[res > 20] = np.nan
    res[res < 0.0] = np.nan
    # print(res)

    d = {'nll':res}
    for iop, op in enumerate(ops):
        d[op] = scan_values[:, iop]

    print((time.perf_counter_ns() - start) / 1e9, "mu s")
    return d





def fit_1d(fout, op1, m1, npoints, **kwargs):
    # op1 = 'cWtil'
    # m1 = 2.5
    # npoints = 1000
    c1 = np.linspace(-m1, m1, npoints)

    start = time.perf_counter_ns()
    # nruns = 100
    # for i in range(nruns):
    res = []

    params = kwargs.pop("params")
    init_pars = kwargs.pop("init_pars")
    nll_bestfit = kwargs.pop("nll_bestfit")

    kwargs["fixed_params"][params.index(op1)] = True

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        tasks = []
        for coeff in c1:
            _init_pars = init_pars.copy()
            tasks.append(pool.submit(fit1d, op1, coeff, params, _init_pars, **kwargs))

        concurrent.futures.wait(tasks)
        res = [k.result() for k in tasks]
    # res = []
    # for coeff in c1:
    #     _init_pars = init_pars.copy()
    #     res.append(fit1d(op1, coeff, _init_pars))

    # for coeff in coeffs:
    #     fit(coeffs)
    res = np.array(res)
    res -= nll_bestfit
    res[res > 20] = np.nan
    res[res < 0.0] = np.nan

    print((time.perf_counter_ns() - start) / 1e9, "mu s")
    # print(res)

    # im = plt.pcolormesh(c1, c2, res.reshape((c1.shape[0], c2.shape[0])), vmin=0.0, vmax=10)
    # print(c1, res)
    # plt.plot(c1, res)
    # # plt.colorbar(im)
    # plt.savefig('test.png')

    # import pickle
    # with open('res.pkl', 'wb') as file:
    #     pickle.dump({op1: c1, 'nll': res}, file)
    fout["limit"] = {op1: c1, "nll": res}

# def scan(**kwargs):

def fit2d(
    op1,
    op2,
    coeff,
    params,
    init_pars,
    **kwargs,
):
    init_pars[params.index(op1)] = coeff[0]
    init_pars[params.index(op2)] = coeff[1]
    # fixed_params[params.index(op1)] = True
    # fixed_params[params.index(op2)] = True
    # res.append(pyhf.infer.mle.twice_nll(init_pars, data, model))
    pdf = kwargs.pop("model")
    data = kwargs.pop("data")
    _fit = pyhf.infer.mle.fit(
        data,
        pdf,
        init_pars=init_pars,
        return_fitted_val=True,
        **kwargs,
    )
    # print(_fit)
    return _fit[1]


def fit_2d(fout, op1, op2, m1=2.5, m2=6.0, npoints=1000, **kwargs):
    # op1 = 'cWtil'
    # op2 = 'cHWtil'
    # m1 = 2.5
    # m2 = 6.0
    # npoints = 1000
    c1 = np.linspace(-m1, m1, int(np.sqrt(npoints)))
    c2 = np.linspace(-m2, m2, int(np.sqrt(npoints)))
    # print(np.meshgrid(c1, c2))

    coeffs = np.dstack(np.meshgrid(c1, c2)).reshape(-1, 2)
    start = time.perf_counter_ns()
    # nruns = 100
    # for i in range(nruns):
    # res = []
    # fixed_params[params.index(op1)] = True
    # fixed_params[params.index(op2)] = True

    # def fit(coeff, init_pars):
    #     init_pars[params.index(op1)] = coeff[0]
    #     init_pars[params.index(op2)] = coeff[1]
    #     # res.append(pyhf.infer.mle.twice_nll(init_pars, data, model))
    #     return pyhf.infer.mle.fit(data, model, init_pars=init_pars, par_bounds=par_bounds, fixed_params=fixed_params, return_fitted_val=True)[1]

    params = kwargs.pop("params")
    init_pars = kwargs.pop("init_pars")
    nll_bestfit = kwargs.pop("nll_bestfit")

    kwargs["fixed_params"][params.index(op1)] = True
    kwargs["fixed_params"][params.index(op2)] = True

    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        tasks = []
        for coeff in coeffs:
            _init_pars = init_pars.copy()
            # tasks.append(pool.submit(fit, coeff, _init_pars))
            tasks.append(pool.submit(fit2d, op1, op2, coeff, params, _init_pars, **kwargs))
        concurrent.futures.wait(tasks)
        res = [k.result() for k in tasks]

    # for coeff in coeffs:
    #     fit(coeffs)
    res = np.array(res)
    res -= nll_bestfit
    # res[res>20] = -1e+8
    _res = res.copy()
    _res[_res > 20] = np.nan
    _res[_res < 0.0] = np.nan

    print((time.perf_counter_ns() - start) / 1e9, "mu s")
    # print(res)

    # im = plt.pcolormesh(c1, c2, _res.reshape((c1.shape[0], c2.shape[0])), vmin=0.0, vmax=15)
    # plt.colorbar(im)
    # plt.savefig('test.png')

    fout["limit"] = {op1: coeffs[:, 0], op2: coeffs[:, 1], "nll": res}
    # sys.exit()


variables = ["mjj", "detajj", "dphijj", "ptj1", "events"]

ops = ["cWtil", "cHWtil"]
limits = [1.0, 2.5]

ops = ["cWtil"]
limits = [1.0]
tot_limits = {
    'cWtil': 0.7,
    'cHBtil': 6.0,
    'cHWBtil': 5.0,
    'cbHIm': 1500.0,
    'cQtQb1Im': 400.0,
    'cbWIm': 43.0,
    'cHGtil': 7.0,
    'cHWtil': 2.0,
    # 'cbBIm': 10.0,
    # 'ctBIm': 10.0,
    'ctWIm': 2.0,
    'cbBIm': 130.0,
    'ctBIm': 5.0,
    "cQtQb8Im": 700.0,
    "cHtbIm": 800.0,
}

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

good_ops = ['cWtil', 'cHWtil', 'cHWBtil', 'ctWIm', 'cHBtil']
# variables = ["mjj"]
# # for ops in list(itertools.combinations(tot_ops, 2)):
# for op in tot_ops:
#     if op != 'cWtil': continue
#     ops = [op]
#     limits = [tot_limits.get(ops[0], 100.0)]

#     # ops = list(ops)
#     # print(ops)
#     # # if 'cbHIm' not in ops:
#     # #     continue
#     # limits = [tot_limits.get(ops[0], 10.0), tot_limits.get(ops[1], 10.0)]
#     # # limits = [2.0, 2.0]

#     npoints = 1000

#     for variable in variables:
#         fout = uproot.recreate(f'results/{variable}_{"_".join(ops)}.root')
#         workspace = read_ws(f"cards/{variable}.json", good_ops)
#         # fit_1d('cHWtil', m1=10, npoints=100)

#         d = scan(ops, limits, npoints, **workspace)
#         fout['limit'] = d
#         fout.close()
#     break

npoints = 10
job = [(['cWtil'], [0.6], 'mjj')]

for ops, limits, variable in job:
    fout = uproot.recreate(f'results/{variable}_{"_".join(ops)}.root')
    workspace = read_ws(f"cards/{variable}.json", good_ops)
    # fit_1d('cHWtil', m1=10, npoints=100)

    d = scan(ops, limits, npoints, **workspace)
    fout['limit'] = d
    fout.close()
# fit_2d('cWtil', 'cHGtil', m1=0.8, m2=4, npoints=500)
# tree = uproot.open('results.root')['limit']
# c1 = tree['cWtil'].array().to_numpy()
# res = tree['nll'].array().to_numpy()
# plot1d(c1, res)

# fit_2d('cWtil', 'ctWIm')
# tree = uproot.open('results.root')['limit']
# c1 = tree['cWtil'].array().to_numpy()
# c2 = tree['ctWIm'].array().to_numpy()
# res = tree['nll'].array().to_numpy()
# plot2d(c1, c2, res)


sys.exit()


import pickle

with open("res.pkl", "rb") as file:
    d = pickle.load(file)
    c1 = d["c1"]
    c2 = d["c2"]
    res = d["res"]
# im = plt.pcolormesh(Z.reshape(x.shape[0], y.shape[0]), vmin=0.0, vmax=10)
im = plt.pcolormesh(c1, c2, res.reshape((c1.shape[0], c2.shape[0])), vmin=0.0, vmax=10)
plt.colorbar(im)
plt.savefig("test1.png")
c1c2 = np.dstack(np.meshgrid(c1, c2)).reshape(-1, 2)
c1 = c1c2[:, 0]
c2 = c1c2[:, 1]
print(len(list(zip(c1, c2))), len(res))


# sys.exit()

# result = pyhf.infer.mle.fit(data, model, init_pars=init_pars, par_bounds=par_bounds, fixed_params=fixed_params, return_uncertainties=True)
# print(result)
