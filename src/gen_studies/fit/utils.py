import os
from textwrap import dedent

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from gen_studies.plot.utils import cmap


def get_datacard_header(bin_name, data_integral):
    return dedent(f"""\n
    ## Shape input card
    imax 1 number of channels
    jmax * number of background
    kmax * number of nuisance parameters
    ----------------------------------------------------------------------------------------------------
    bin         {bin_name}
    observation {data_integral}
    shapes  *           * shapes.root     histo_$PROCESS histo_$PROCESS_$SYSTEMATIC
    shapes  data_obs           * shapes.root     histo_Data
    """)


def make_datacard(
    input_file,
    region,
    variable,
    systematics,
    structure_name,
    structures,
):
    output_path = f"datacards/{structure_name}/{region}/{variable}"

    os.makedirs(output_path, exist_ok=True)
    output_file = uproot.recreate(f"{output_path}/shapes.root")

    structure = structures[structure_name]

    sig_idx = 0
    bkg_idx = 1

    bin_name = f"{region}_{variable}"
    rows = [
        ["bin"],
        ["process"],
        ["process"],
        ["rate"],
    ]
    systs = {}

    h_data = 0
    for sample_name in structure:
        final_name = f"{region}/{variable}/histo_{sample_name}"
        h = input_file[final_name].to_hist().copy()
        name = structure[sample_name]["name"]
        isSignal = structure[sample_name]["isSignal"]
        isData = structure[sample_name]["isData"]
        noStat = structure[sample_name].get("noStat", False)

        if isSignal:
            idx = sig_idx
            sig_idx -= 1
        else:
            idx = bkg_idx
            bkg_idx += 1

        if isData:
            h_data = h.copy()

        if isData and name != "Data":
            raise Exception("Cannot use isData with a name != 'Data'")

        if noStat:
            histo_view = h.view(True)
            histo_view.variance = np.zeros_like(histo_view.variance)

        rows[0].append(bin_name)
        rows[1].append(name)
        rows[2].append(str(idx))
        rows[3].append(str(np.sum(h.values(True))))

        for systematic in systematics:
            if sample_name in systematics[systematic]["samples"]:
                if systematics[systematic]["type"] == "lnN":
                    syst = systematics[systematic]["samples"][sample_name]
                else:
                    syst = "1.0"
                    for tag in ["Up", "Down"]:
                        _final_name = final_name + f"_{systematic}{tag}"
                        _h = input_file[_final_name].to_hist().copy()
                        output_file[f"histo_{name}_{systematic}{tag}"] = _h
            else:
                syst = "-"
            print(sample_name, systematic, syst)
            if systematic not in systs:
                systs[systematic] = [systematics[systematic]["type"], syst]
            else:
                systs[systematic].append(syst)

        output_file[f"histo_{name}"] = h

    if isinstance(h_data, int):
        h_data = h.copy()
        histo_view = h_data.view(True)
        histo_view.value = np.zeros_like(histo_view.value)
        histo_view.variance = np.zeros_like(histo_view.variance)
        output_file["histo_Data"] = h_data
    datacard = get_datacard_header(bin_name, np.sum(h_data.values(True)))
    for row in rows:
        datacard += "\t".join(row) + "\n"

    datacard += "-" * 100 + "\n"

    for syst in systs:
        datacard += systematics[syst]["name"] + "\t" + "\t".join(systs[syst]) + "\n"
    with open(f"{output_path}/datacard.txt", "w") as file:
        file.write(datacard)


def plot1d(file, ops, lumi):
    op = ops[0]
    x = file["limit"][f"k_{op}"].array().to_numpy()
    y = file["limit"]["deltaNLL"].array().to_numpy()

    y -= y[0]

    x = x[1:]
    y = y[1:]

    mask = (y >= 0.0) & (y <= 10.0)

    x = x[mask]
    y = y[mask]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 10),
        dpi=100,
    )
    fig.tight_layout(pad=-0.5)

    hep.cms.label("Work in progress", data=False, ax=ax, exp="", lumi=str(lumi))

    target = 1.0
    for target in [1.0, 3.84]:
        xmins = x[np.where(np.diff(np.sign(y - np.tile(target, y.shape))))]
        ax.text(
            xmins[0],
            target,
            str(round(xmins[0], 2)),
            ha="left",
        )
        ax.text(
            xmins[1],
            target,
            str(round(xmins[1], 2)),
            ha="right",
        )
    ax.plot(x, y, color="black")
    ax.plot(x, np.ones_like(x) * 1, color="red", linestyle="dashed")
    ax.text(
        x[0],
        1.0,
        str("68%"),
        ha="left",
    )
    ax.plot(x, np.ones_like(x) * 3.84, color="green", linestyle="dashed")
    ax.text(
        x[0],
        3.84,
        str("95%"),
        ha="left",
    )
    ax.set_ylabel("$-2\\Delta LL$", loc="top")
    ax.set_xlabel(op, loc="right")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, None)
    ax.grid(which="both")
    ax.legend()
    fig.savefig(
        f"scans/scan_1d_{op}.png",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )
    plt.close()


def plot2d(
    file,
    ops,
    lumi,
):
    debug = True
    c1 = file["limit"][f"k_{ops[0]}"].array().to_numpy()
    c2 = file["limit"][f"k_{ops[1]}"].array().to_numpy()
    res = file["limit"]["deltaNLL"].array().to_numpy()
    res -= res[0]
    c1 = c1[1:]
    c2 = c2[1:]
    res = res[1:]

    x = np.unique(c1)
    y = np.unique(c2)

    Z = (np.ones(shape=(x.shape[0], y.shape[0])) * 1000.0).reshape(-1, 1)
    # a lot of tweaks
    centers = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
    arr1 = centers[:]
    arr2 = np.zeros((c1.shape[0], 2))
    arr2[:, 0] = c1
    arr2[:, 1] = c2
    assert arr1.shape[1] == arr2.shape[1]
    cols1 = arr1.shape[1]
    cols2 = arr2.shape[1]
    dt1 = {
        "names": ["f{}".format(i) for i in range(cols1)],
        "formats": cols1 * [arr1.dtype],
    }
    dt2 = {
        "names": ["f{}".format(i) for i in range(cols2)],
        "formats": cols2 * [arr2.dtype],
    }

    _, ind1, ind2 = np.intersect1d(arr1.view(dt1), arr2.view(dt2), return_indices=True)
    Z[ind1] = res[ind2].reshape(-1, 1)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, 10),
        dpi=100,
    )
    fig.tight_layout(pad=-0.5)

    hep.cms.label("Work in progress", data=False, ax=ax, exp="", lumi=str(lumi))

    if debug:
        im = ax.pcolormesh(x, y, Z.reshape(x.shape[0], y.shape[0]), vmin=0.0, vmax=100)
        fig.colorbar(im, ax=ax)

    color = cmap[1]

    def fmt(x):
        if x == 2.0:
            return "68%"
        elif x == 5.99:
            return "95%"

    CS = ax.contour(
        Z.reshape(x.shape[0], y.shape[0]),
        extent=[x[0], x[-1], y[0], y[-1]],
        levels=[2],
        colors=color,
    )
    ax.clabel(
        CS,
        CS.levels,
        inline=True,
        fmt=fmt,
        fontsize=20,
        colors=["red"],
    )
    CS = ax.contour(
        Z.reshape(x.shape[0], y.shape[0]),
        extent=[x[0], x[-1], y[0], y[-1]],
        levels=[5.99],
        colors=color,
        linestyles="dashed",
    )
    ax.clabel(
        CS,
        CS.levels,
        inline=True,
        fmt=fmt,
        fontsize=20,
        colors=["red"],
    )
    # SM
    ax.plot(
        [0],
        [0],
        marker="*",
        markersize=10,
        label="SM",
        color="red",
    )

    # Lines at 0
    ax.plot(
        x,
        np.zeros_like(x),
        color="black",
        linestyle="dashed",
    )
    ax.plot(
        np.zeros_like(y),
        y,
        color="black",
        linestyle="dashed",
    )

    ax.grid(which="both")

    ax.set_xlabel(ops[0], loc="right")
    ax.set_ylabel(ops[1], loc="top")
    ax.legend()
    fig.savefig(
        f"scans/scan_2d_{ops[0]}_{ops[1]}.png",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )
    plt.close()
