import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

d = hep.style.CMS
plt.style.use([d, hep.style.firamath])

cmap = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
sm_color = cmap[0]
lin_color = cmap[1]
quad_color = cmap[2]
darker_factor = 4 / 5


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

    err = np.square(ratio) * (
        np.power(err1 / cont1, 2) + np.power(err2 / cont2, 2)
    )

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


def plot_ratio_single_err(h1, h2, ax, label, color, **kwargs):
    centers = h1.axes[0].centers
    edges = h1.axes[0].edges
    cont1 = h1.values()
    cont2 = h2.values()
    ratio = cont1 / cont2
    err1 = np.sqrt(h1.variances())

    err = err1 / cont2

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


def format_variable_name(variable):
    variable = variable + ""
    formatted = "$" + variable + "$"
    formatted = formatted.replace("eta", r"\eta")
    formatted = formatted.replace("phi", r"\phi")
    formatted = formatted.replace("pt", "p^T")

    if variable.startswith("d"):
        formatted = formatted.replace("d", r"\Delta", 1)
    formatted = formatted.replace(
        formatted[-3:-1], "_{" + formatted[-3:-1] + "}"
    )

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
    plot_dict,
    variable,
    op,
    scale,
    formatted,
    lumi,
    plot_ylim_ratio=None,
):
    # Begin single plot
    # plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 10),
        dpi=100,
    )
    fig.tight_layout(pad=-0.5)

    hep.cms.label(
        "Work in progress", data=False, ax=ax[0], exp="", lumi=str(lumi)
    )

    hlast = 0
    hlast_bkg = 0
    for isample, sample_name in enumerate(plot_dict):
        # print(sample_name)
        isSignal = plot_dict[sample_name].get("isSignal", False)

        color = plot_dict[sample_name].get("color", "black")
        h_sm = input_file[variable][f"histo_{sample_name}"].to_hist()
        if isinstance(hlast, int):
            hlast = h_sm.copy()
        else:
            hlast += h_sm.copy()
        if not isSignal:
            if isinstance(hlast_bkg, int):
                hlast_bkg = h_sm.copy()
            else:
                hlast_bkg += h_sm.copy()
        label = plot_dict[sample_name]["name"]
        centers = hlast.axes[0].centers
        edges = hlast.axes[0].edges
        content = hlast.values()
        integral = "{:.3e}".format(np.sum(h_sm.values()))

        if isSignal:
            color = get_lighter_color(color)
        if plot_dict[sample_name].get("superimposed", False):
            _color = get_darker_color(color)
            # _color = color
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
                yerr=np.sqrt(h_sm.variances()),
                fmt="o",
                label=label + " imposed",
                color=_color,
                markersize=0.1,
            )

        kwargs = dict(fill=True, zorder=-isample)
        if isSignal:
            kwargs = dict(fill=False, zorder=+isample)
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
    err = np.sqrt(hlast_bkg.variances())
    ax[0].stairs(
        content + err,
        edges,
        baseline=content - err,
        fill=True,
        zorder=+10,
        hatch="///",
        color="darkgrey",
        alpha=1,
        facecolor="none",
        linewidth=0.0,
    )

    # h_lin = input_file[variable][f"histo_{signal}_lin_{op}"]
    # h_quad = input_file[variable][f"histo_{signal}_quad_{op}"]

    # plot_good(h_sm, ax[0], "SM", sm_color)
    # plot_good(h_lin, ax[0], f"Lin {op}", lin_color)
    # plot_good(h_quad, ax[0], f"Quad {op}", quad_color)

    edges = h_sm.axes[0].edges
    ax[0].plot(edges, np.zeros_like(edges), color="black")

    # # Setup legend alignment
    # plt.rcParams["font.monospace"] = ["Fira Mono"]
    # legend = ax[0].legend(prop={"family": "monospace", "weight": "semibold"})

    # max_len1 = 0
    # max_len2 = 0
    # for t in legend.get_texts():
    #     text = t.get_text().split("[")
    #     max_len1 = max(max_len1, len(text[0]))
    #     max_len2 = max(max_len2, len(text[1]))
    # for t in legend.get_texts():
    #     text = t.get_text().split("[")
    #     new_text = "{:<" + str(max_len1) + "}- [{:>" + str(max_len2) + "}"
    #     new_text = new_text.format(text[0], text[1])
    #     t.set_text(new_text)
    ax[0].legend()

    # Setup labels and scale
    ax[0].set_ylabel("Events")
    if scale == "log":
        ax[0].set_yscale(scale)

    # Lower plot
    ax[1].plot(edges, np.zeros_like(edges), color="black")

    content = hlast_bkg.values()
    err = np.sqrt(hlast_bkg.variances())

    ax[1].stairs(
        (content + err) / content - 1,
        edges,
        baseline=(content - err) / content - 1,
        fill=True,
        color="lightgrey",
    )

    for isample, sample_name in enumerate(plot_dict):
        # print(sample_name)
        isSignal = plot_dict[sample_name].get("isSignal", False)
        if not isSignal:
            continue
        label = plot_dict[sample_name]["name"]
        h_sm = input_file[variable][f"histo_{sample_name}"].to_hist().copy()
        color = plot_dict[sample_name].get("color", "black")
        plot_ratio_single_err(
            h_sm,
            hlast_bkg,
            ax[1],
            label,
            color,
            **dict(stairs=dict(baseline=0.0), plot_errors=True),
        )

    ax[1].legend()

    ax[1].set_xlim(edges[0], edges[-1])
    # min_val_ratio = -0.1
    # max_val_ratio = 0.5
    if plot_ylim_ratio:
        ax[1].set_ylim(*plot_ylim_ratio)

    ax[1].set_ylabel("Component / SM")
    ax[1].set_xlabel(formatted)

    fig.savefig(
        f"plots/{scale}_{op}_{variable}.png",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )
    plt.close()
