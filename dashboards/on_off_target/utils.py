import matplotlib.pyplot as plt


def make_line_plot(ax, x, y, err=None, fill=False, **kwargs):
    dic = dict(
        color="black",
        marker=None,
        markersize=3,
        linestyle="-",
        linewidth=1.,
        label=None
    )
    dic.update(kwargs)
    if err is None:
        ax.plot(x,y,**dic)
    else:
        if fill:
            ax.plot(x, y, **dic)
            ax.fill_between(x, y - err, y + err, color=dic["color"], alpha=0.2)
        else:
            ax.errorbar(x, y, yerr=err, ecolor=dic["color"],capsize=2,capthick=None,**dic)

