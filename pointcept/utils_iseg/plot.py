import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_curves(
    x,
    y_dict_list,
    color_map,
    linestyle_map,
    xlabel="",
    ylabel="",
    title="",
    save_path=None,
):
    """"""

    plt.figure(figsize=(8, 6))
    for y_dict in y_dict_list:
        plt.plot(
            x,
            y_dict["y"],
            color=color_map[y_dict["color_name"]],
            linestyle=linestyle_map[y_dict["linestyle_name"]],
        )

    # 构造 legend 句柄
    color_title = color_map.pop("title")
    handles = [Line2D([], [], color="none", label=color_title)]
    labels = [color_title]
    for color_name, color_value in color_map.items():
        handles.append(Line2D([], [], color=color_value, lw=2, label=color_name))
        labels.append(color_name)
    # 优先使用color作为区分, 如果 linestyle 只有一种就不列出
    linestyle_title = linestyle_map.pop("title")
    if len(linestyle_map) > 1:
        handles.append(Line2D([], [], color="none", label=linestyle_title))
        labels.append(linestyle_title)
        for linestyle_name, linestyle_value in linestyle_map.items():
            handles.append(
                Line2D(
                    [],
                    [],
                    color="black",
                    lw=2,
                    linestyle=linestyle_value,
                    label=linestyle_name,
                )
            )
            labels.append(linestyle_name)
    plt.legend(handles=handles, labels=labels, loc="best", frameon=True, handlelength=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000)
    plt.show()


if __name__ == "__main__":
    color_map = {
        "mAP": "#1f77b4",  # 蓝色
        "mAP@50": "#ff7f0e",  # 橙色
        "mAP@25": "#2ca02c",  # 绿色
    }
    linestyle_map = {
        "100 trained queries": "-",
        "200 trained queries": "--",
    }
