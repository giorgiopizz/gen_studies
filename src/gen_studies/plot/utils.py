from copy import deepcopy

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

cmap = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
sm_color = cmap[0]
lin_color = cmap[1]
quad_color = cmap[2]
darker_factor = 4 / 5


def set_plot_style():
    d = hep.style.CMS
    plt.style.use([d, hep.style.firamath])


def set_plot_header(plot_label, ax, lumi):
    hep.cms.label(plot_label, data=True, ax=ax, exp="", lumi=str(lumi))


def plot_good(h, ax, label, color, **kwargs):
    centers = h.axes[0].centers
    edges = h.axes[0].edges
    content = h.values()
    sumw2 = h.variances()
    integral = "{:.3e}".format(np.sum(content))

    ax.errorbar(
        centers,
        content,
        yerr=np.sqrt(sumw2),
        fmt="o",
        label=label + f" [{integral}]",
        color=color,
        **kwargs.get("errorbar", {}),
    )
    ax.stairs(
        content,
        edges,
        color=color,
        **kwargs.get("stairs", {}),
    )


def plot_ratio(h1, h2, ax, label, color, **kwargs):
    centers = h1.axes[0].centers
    edges = h1.axes[0].edges
    cont1 = h1.values()
    cont2 = h2.values()
    ratio = cont1 / cont2
    err1 = h1.variances()
    err2 = h2.variances()

    err = np.square(ratio) * (np.power(err1 / cont1, 2) + np.power(err2 / cont2, 2))

    if kwargs.get("plot_errors", True):
        ax.errorbar(
            centers,
            ratio,
            yerr=err,
            fmt="o",
            label=label,
            color=color,
            **kwargs.get("errorbar", {}),
        )

    ax.stairs(
        ratio,
        edges,
        color=color,
        **kwargs.get("stairs", {}),
    )


def plot_ratio_single_err(
    h1,
    err1_do,
    err1_up,
    h2,
    ax,
    label,
    color,
    **kwargs,
):
    centers = h1.axes[0].centers
    edges = h1.axes[0].edges
    cont1 = h1.values()
    cont2 = h2.values()
    ratio = cont1 / cont2
    # err1 = np.sqrt(h1.variances())

    err_up = err1_up / cont2
    err_do = err1_do / cont2

    if kwargs.get("plot_errors", True):
        ax.errorbar(
            centers,
            ratio,
            yerr=(err_do, err_up),
            fmt="o",
            label=label,
            color=color,
            **kwargs.get("errorbar", {}),
        )

    ax.stairs(
        ratio,
        edges,
        color=color,
        **kwargs.get("stairs", {}),
    )


def format_variable_name(variable):
    variable = variable + ""
    formatted = "$" + variable + "$"
    formatted = formatted.replace("eta", r"\eta")
    formatted = formatted.replace("phi", r"\phi")
    formatted = formatted.replace("pt", "p^T")

    if variable.startswith("d"):
        formatted = formatted.replace("d", r"\Delta", 1)
    formatted = formatted.replace(formatted[-3:-1], "_{" + formatted[-3:-1] + "}")

    return formatted


def get_darker_color(color):
    if not isinstance(color, tuple):
        rgb = list(mpl.colors.to_rgba(color)[:-1])
    else:
        rgb = list(color)

    rgb[0] = rgb[0] * darker_factor
    rgb[1] = rgb[1] * darker_factor
    rgb[2] = rgb[2] * darker_factor
    return tuple(rgb)


def get_lighter_color(color):
    rgb = list(mpl.colors.to_rgba(color)[:-1])
    lighter_factor = 1.0 / darker_factor
    rgb[0] = rgb[0] * lighter_factor
    rgb[1] = rgb[1] * lighter_factor
    rgb[2] = rgb[2] * lighter_factor
    return tuple(rgb)


def final_bkg_plot(
    input_file,
    samples,
    plot_name,
    plots,
    region,
    variable,
    systematics,
    scale,
    formatted,
    lumi,
    plot_label,
    plot_ylim_ratio=None,
):
    # Begin single plot
    set_plot_style()
    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 10),
        dpi=100,
    )
    fig.tight_layout(pad=-0.5)

    set_plot_header(plot_label, ax[0], lumi)

    hlast = 0
    hlast_bkg = 0
    v_syst_bkg = {}
    v_syst_sig = {}
    plot_dict = deepcopy(plots[plot_name])
    for isample, sample_name in enumerate(plot_dict):
        isSignal = plot_dict[sample_name].get("isSignal", False)

        color = plot_dict[sample_name].get("color", "black")
        label = plot_dict[sample_name]["name"]
        final_name = f"{region}/{variable}/histo_{sample_name}"
        h_sm = input_file[final_name].to_hist()

        if isinstance(hlast, int):
            hlast = h_sm.copy()
        else:
            hlast += h_sm.copy()
        if not isSignal:
            if isinstance(hlast_bkg, int):
                hlast_bkg = h_sm.copy()
            else:
                hlast_bkg += h_sm.copy()

        # include systematics+stat
        for syst in list(systematics.keys()) + ["stat"]:
            if syst == "stat":
                vvar_up = h_sm.values() + np.sqrt(h_sm.variances())
                vvar_do = h_sm.values() - np.sqrt(h_sm.variances())
            else:
                if sample_name not in systematics[syst]["samples"]:
                    vvar_up = h_sm.values().copy()
                    vvar_do = h_sm.values().copy()
                else:
                    syst_type = systematics[syst]["type"]
                    if syst_type == "shape":
                        _final_name = final_name + f"_{syst}Up"
                        vvar_up = input_file[_final_name].values()

                        _final_name = final_name + f"_{syst}Down"
                        vvar_do = input_file[_final_name].values()
                    elif syst_type == "lnN":
                        scaling = float(systematics[syst]["samples"][sample_name])
                        vvar_up = scaling * h_sm.values()
                        vvar_do = 1.0 / scaling * h_sm.values()

            if not isSignal:
                if syst not in v_syst_bkg:
                    v_syst_bkg[syst] = {
                        "up": vvar_up.copy(),
                        "do": vvar_do.copy(),
                    }
                else:
                    v_syst_bkg[syst]["up"] += vvar_up
                    v_syst_bkg[syst]["do"] += vvar_do
            else:
                vvar_up = np.square(vvar_up - h_sm.values())
                vvar_do = np.square(vvar_do - h_sm.values())
                if sample_name not in v_syst_sig:
                    v_syst_sig[sample_name] = {
                        "up": vvar_up.copy(),
                        "do": vvar_do.copy(),
                    }
                else:
                    v_syst_sig[sample_name]["up"] += vvar_up
                    v_syst_sig[sample_name]["do"] += vvar_do
        print(v_syst_sig)

        if isSignal:
            v_syst_sig[sample_name]["up"] = np.sqrt(v_syst_sig[sample_name]["up"])
            v_syst_sig[sample_name]["do"] = np.sqrt(v_syst_sig[sample_name]["do"])

        centers = hlast.axes[0].centers
        edges = hlast.axes[0].edges
        content = hlast.values()
        integral = "{:.3e}".format(np.sum(h_sm.values()))

        if isSignal:
            color = get_lighter_color(color)
        if plot_dict[sample_name].get("superimposed", False):
            _color = get_darker_color(color)
            if not isSignal:
                raise Exception("Can only plot superimposed for signals!")

            ax[0].stairs(
                h_sm.values(),
                edges,
                color=_color,
                # label=label + f" [{integral}]",
                fill=False,
                zorder=+isample,
                linewidth=2,
            )
            ax[0].errorbar(
                centers,
                h_sm.values(),
                yerr=(
                    v_syst_sig[sample_name]["do"],
                    v_syst_sig[sample_name]["up"],
                ),
                fmt="o",
                label=label + " imposed",
                color=_color,
                markersize=0.1,
            )

        kwargs = dict(fill=True, zorder=-isample)
        if isSignal:
            kwargs = dict(fill=False, zorder=+isample)
        if plot_dict[sample_name].get("stacked", True):
            ax[0].stairs(
                content,
                edges,
                color=color,
                label=label + f" [{integral}]",
                edgecolor=get_darker_color(color),
                linewidth=2,
                **kwargs,
            )
    content = hlast_bkg.values()

    # Compute squared sum of different systematics
    vvar_up = 0
    vvar_do = 0
    for syst in v_syst_bkg:
        if isinstance(vvar_up, int):
            vvar_up = np.square(v_syst_bkg[syst]["up"].copy() - content)
            vvar_do = np.square(v_syst_bkg[syst]["do"].copy() - content)
        else:
            vvar_up += np.square(v_syst_bkg[syst]["up"].copy() - content)
            vvar_do += np.square(v_syst_bkg[syst]["do"].copy() - content)

    vvar_up = np.sqrt(vvar_up)
    vvar_do = np.sqrt(vvar_do)

    ax[0].stairs(
        content + vvar_up,
        edges,
        baseline=content - vvar_do,
        fill=True,
        zorder=+10,
        hatch="///",
        color="darkgrey",
        alpha=1,
        facecolor="none",
        linewidth=0.0,
    )

    edges = h_sm.axes[0].edges
    ax[0].plot(edges, np.zeros_like(edges), color="black")

    ax[0].legend()

    # Setup labels and scale
    ax[0].set_ylabel("Events")
    if scale == "log":
        ax[0].set_yscale(scale)

    # Lower plot
    ax[1].plot(edges, np.zeros_like(edges), color="black")

    content = hlast_bkg.values()

    ax[1].stairs(
        (content + vvar_up) / content - 1,
        edges,
        baseline=(content - vvar_do) / content - 1,
        fill=True,
        color="lightgrey",
    )

    for isample, sample_name in enumerate(plot_dict):
        # print(sample_name)
        isSignal = plot_dict[sample_name].get("isSignal", False)
        if not isSignal:
            continue
        label = plot_dict[sample_name]["name"]
        h_sm = input_file[region][variable][f"histo_{sample_name}"].to_hist().copy()
        color = plot_dict[sample_name].get("color", "black")
        plot_ratio_single_err(
            h_sm,
            v_syst_sig[sample_name]["do"],
            v_syst_sig[sample_name]["up"],
            hlast_bkg,
            ax[1],
            label,
            color,
            **dict(stairs=dict(baseline=0.0), plot_errors=True),
        )

    ax[1].legend()

    ax[1].set_xlim(edges[0], edges[-1])
    if plot_ylim_ratio:
        ax[1].set_ylim(*plot_ylim_ratio)

    ax[1].set_ylabel("Component / SM")
    ax[1].set_xlabel(formatted)

    fig.savefig(
        f"plots/{scale}_{plot_name}_{region}_{variable}.png",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )
    plt.close()
