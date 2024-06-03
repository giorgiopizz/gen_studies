import itertools

import awkward as ak
import hist
import numpy as np


def add_dict(d1, d2):
    if isinstance(d1, dict):
        d = {}
        common_keys = set(list(d1.keys())).intersection(list(d2.keys()))
        for key in common_keys:
            d[key] = add_dict(d1[key], d2[key])
        for key in d1:
            if key in common_keys:
                continue
            d[key] = d1[key]
        for key in d2:
            if key in common_keys:
                continue
            d[key] = d2[key]

        return d
    elif isinstance(d1, np.ndarray):
        return np.concatenate([d1, d2])
    elif isinstance(d1, ak.highlevel.Array):
        return ak.concatenate([d1, d2])
    else:
        return d1 + d2


def add_dict_iterable(iterable):
    tmp = -99999
    for it in iterable:
        if tmp == -99999:
            tmp = it
        else:
            tmp = add_dict(tmp, it)
    return tmp


def read_ops(filename):
    with open(filename) as file:
        lines = file.read().split("\n")
    lines = list(
        filter(lambda k: k.startswith("#"), lines)
    )  # only take comments and not mixed terms (',')
    rwgts = {}
    ops = []

    for irwgt, line in enumerate(lines):
        if "sm" in line.lower():
            _, op, _ = line.split(" ")
            rwgts["sm"] = irwgt
        elif "," not in line:
            _, op_val, _ = line.split(" ")
            op, _ = op_val.split("=")
            ops.append(op)
            # res.append([op, val, rwgt])
            rwgts[op_val] = irwgt
        else:
            splitted = line.split(" ")
            ops_val = " ".join(splitted[1:-1])
            # res.append([op, val, rwgt])
            rwgts[ops_val] = irwgt
    return list(set(ops)), rwgts


def create_components(events, active_ops, rwgts):
    weights = events["LHEReweightingWeight"]
    new_weights = {}
    new_weights["sm"] = ak.copy(weights[:, rwgts["sm"]])
    for op in active_ops:
        # make sm_lin_quad
        new_weights[f"sm_lin_quad_{op}"] = ak.copy(weights[:, rwgts[f"{op}=1"]])
        # make linear
        new_weights[f"lin_{op}"] = 0.5 * (
            weights[:, rwgts[f"{op}=1"]] - weights[:, rwgts[f"{op}=-1"]]
        )
        # make quad
        new_weights[f"quad_{op}"] = 0.5 * (
            weights[:, rwgts[f"{op}=1"]]
            + weights[:, rwgts[f"{op}=-1"]]
            - 2 * new_weights["sm"]
        )
    for op1, op2 in list(itertools.combinations(active_ops, 2)):
        _op1, _op2 = op1, op2
        rwgt_key = f"{op1}=1, {op2}=1"
        if rwgt_key not in rwgts:
            rwgt_key = f"{op2}=1, {op1}=1"
            _op1, _op2 = op2, op1
        new_weights[f"mixed_{_op1}_{_op2}"] = (
            weights[:, rwgts[rwgt_key]]
            - new_weights["sm"]
            - new_weights[f"lin_{op1}"]
            - new_weights[f"quad_{op1}"]
            - new_weights[f"lin_{op2}"]
            - new_weights[f"quad_{op2}"]
        )
    events["components"] = ak.zip(new_weights)
    return events


def hist_move_content(h, ifrom, ito):
    """
    Moves content of a histogram from `ifrom` bin to `ito` bin.
    Content and sumw2 of bin `ito` will be the sum of the original `ibin`
    and `ito`.
    Content and sumw2 of bin `ifrom` will be 0.
    Modifies in place the histogram.

    Parameters
    ----------
    h : hist
        Histogram
    ifrom : int
        the index of the bin where content will be reset
    ito : int
        the index of the bin where content will be the sum
    """
    dimension = len(h.axes)
    # numpy view is a numpy array containing two keys, value
    # and variances for each bin
    numpy_view = h.view(True)
    content = numpy_view.value
    sumw2 = numpy_view.variance

    if dimension == 1:
        content[ito] += content[ifrom]
        content[ifrom] = 0.0

        sumw2[ito] += sumw2[ifrom]
        sumw2[ifrom] = 0.0

    elif dimension == 2:
        content[ito, :] += content[ifrom, :]
        content[ifrom, :] = 0.0
        content[:, ito] += content[:, ifrom]
        content[:, ifrom] = 0.0

        sumw2[ito, :] += sumw2[ifrom, :]
        sumw2[ifrom, :] = 0.0
        sumw2[:, ito] += sumw2[:, ifrom]
        sumw2[:, ifrom] = 0.0

    elif dimension == 3:
        content[ito, :, :] += content[ifrom, :, :]
        content[ifrom, :, :] = 0.0
        content[:, ito, :] += content[:, ifrom, :]
        content[:, ifrom, :] = 0.0
        content[:, :, ito] += content[:, :, ifrom]
        content[:, :, ifrom] = 0.0

        sumw2[ito, :, :] += sumw2[ifrom, :, :]
        sumw2[ifrom, :, :] = 0.0
        sumw2[:, ito, :] += sumw2[:, ifrom, :]
        sumw2[:, ifrom, :] = 0.0
        sumw2[:, :, ito] += sumw2[:, :, ifrom]
        sumw2[:, :, ifrom] = 0.0


def hist_fold(h, fold_method):
    """
    Fold a histogram (hist object)

    Parameters
    ----------
    h : hist
        Histogram to fold, will be modified in place (aka no copy)
    fold_method : int
        choices 0: no fold
        choices 1: fold underflow
        choices 2: fold overflow
        choices 3: fold both underflow and overflow
    """
    if fold_method == 1 or fold_method == 3:
        hist_move_content(h, 0, 1)
    if fold_method == 2 or fold_method == 3:
        hist_move_content(h, -1, -2)


def hist_unroll(h):
    """
    Unrolls n-dimensional histogram

    Parameters
    ----------
    h : hist
        Histogram to unroll

    Returns
    -------
    hist
        Unrolled 1-dimensional histogram
    """
    dimension = len(h.axes)
    if dimension != 2:
        raise Exception(
            "Error in hist_unroll: can only unroll 2D histograms, while got ",
            dimension,
            "dimensions",
        )

    numpy_view = h.view()  # no under/overflow!
    nx = numpy_view.shape[0]
    ny = numpy_view.shape[1]
    h_unroll = hist.Hist(
        hist.axis.Regular(nx * ny, 0, nx * ny), hist.storage.Weight()
    )

    numpy_view_unroll = h_unroll.view()
    numpy_view_unroll.value = numpy_view.value.flatten()
    numpy_view_unroll.variance = numpy_view.variance.flatten()

    return h_unroll
