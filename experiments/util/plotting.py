from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap

from util import stretch_constant_interval


class Plotter:
    """
    Plotting utility class for our trajectories. Can generate connected line plots or scatter plots and is extendable
    """

    def __init__(self, threshold, filetype):
        """
        :param threshold: The threshold at which stroke value in the trajectory we should consider a new stroke started
        :param filetype: The filetype all saved figures should have
        """
        self.threshold = threshold
        self.filetype = filetype
        self.actions = {
            "line": self._line,
            "scatter": self._scatter
        }

    def stack(self, params, actions, path=None, show=False, transparent=True):
        """
        Stacks multiple plots on top of each other
        :param params: A list of form:
        [...
        (required parameter tuple, optional parameter dict)
        ...]
        with required parameter tuple typically being (x, y, p, s), each a numpy array of size n
        and the optional parameter dict depending on the function, eg.
        {"size": 100, "color": plt.get_cmap("viridis"), "marker": "o"}
        :param actions: What plotting function to apply to each parameter, string in the self.actions dict
        :param path: Path to save file in
        :param show: Whether to show figure with plt.show() first
        :param transparent: Whether the background of the figure should be transparent
        :return:
        """
        if actions is not list:
            actions = [actions]
        for (param, action) in zip(params, cycle(actions)):
            func = self.actions[action]
            func(plt.gca(), *(param[0]), **(param[1]))
        if path is not None:
            plt.savefig(path + f".{self.filetype}", dpi=300, transparent=transparent)
        if show is not False:
            plt.show()
        plt.close()

    def table(self, rows, cols, params, actions, path=None, show=False, row_labels=None, col_labels=None, transparent=True):
        """
        Puts multiple plots side by side in a table
        :param rows: Number of rows
        :param cols: Number of columns
        :param params: A list of form:
        [...
        (required parameter tuple, optional parameter dict)
        ...]
        with required parameter tuple typically being (x, y, p, s), each a numpy array of size n
        and the optional parameter dict depending on the function, eg.
        {"size": 100, "color": plt.get_cmap("viridis"), "marker": "o"}
        :param actions: What plotting function to apply to each parameter, string in the self.actions dict
        :param path: Path to save file in
        :param show: Whether to show figure with plt.show() first
        :param row_labels: List of strings to label rows with, length needs to be equal to rows
        :param col_labels: List of strings to label columns with, length needs to be equal to cols
        :param transparent: Whether the background of the figure should be transparent
        :return:
        """
        if actions is not list:
            actions = [actions]
        fig, axe = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 3))
        if rows == 1 and cols == 1:
            ax = axe
            axe = np.empty((1,1), dtype=object)
            axe[0][0] = ax

        axes = axe.flatten()
        for k, (param, action) in enumerate(zip(params, cycle(actions))):
            ax = axes[k]
            func = self.actions[action]
            func(ax, *(param[0]), **(param[1]))
        for k in range(len(params), len(axes)):
            ax = axes[k]
            ax.set_visible(False)

        if col_labels is not None:
            for ax, col in zip(axe[0], col_labels):
                ax.set_title(col)

        if row_labels is not None:
            for ax, row in zip(axe[:, 0], row_labels):
                ax.set_ylabel(row, rotation=0, size='large')

        fig.tight_layout()
        if path is not None:
            plt.savefig(path + f".{self.filetype}", dpi=300, transparent=transparent)
        if show is not False:
            plt.show()
        plt.close()

    def _scatter(self, ax, x, y, p, s, size=100, color="blue", marker="o", tight=False):
        """
        Scatterplots a trajectory
        :param ax: The matplotlib axis object to plot onto
        :param x: Numpy array of x positions
        :param y: Numpy array of y positions
        :param p: Numpy array of pen pressure
        :param s: Numpy array of "stroke-start" feature
        :param size: Size of the scattered dots
        :param color: Color of the scattered dots, can be a Colormap instance, e.g. for using gradients
        :param marker: Shape of the colored dots
        :param tight: Whether to plot without margins
        :return:
        """
        # gradients
        cmap = None
        if isinstance(color, Colormap):
            c = np.linspace(0, 1, len(x))
            cmap = color
        else:
            c = color

        b = size
        ax.scatter(x, y, s=(b * p + 10e-4), c=c, marker=marker, cmap=cmap)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        if tight:
            x_mid = (x.min() + x.max()) / 2
            x_breadth = abs(x.min()) + abs(x.max())
            y_mid = (y.min() + y.max()) / 2
            y_breadth = abs(y.min()) + abs(y.max())
            breadth = max(x_breadth, y_breadth)
            ax.set_xlim(x_mid - breadth/2 - 0.1, x_mid + breadth/2 + 0.1)
            ax.set_ylim(y_mid - breadth/2 - 0.1, y_mid + breadth/2 + 0.1)
        else:
            ax.autoscale()
        lim = max(x.max(), abs(x.max()), y.max(), abs(y.max()))
        ax.set_aspect(1)

    def _line(self, ax, x, y, p, s, color="blue", pressure_scale=1, tight=False, background=None, stretch_cmap=None):
        """
        Creates a lineplot from a trajectory, slightly hacked together with a LineCollection.
        :param ax: The matplotlib axis object to plot onto
        :param x: Numpy array of x positions
        :param y: Numpy array of y positions
        :param p: Numpy array of pen pressure
        :param s: Numpy array of "stroke-start" feature
        :param color: Color of the scattered dots, can be a Colormap instance, e.g. for using gradients
        :param pressure_scale: How much the thickness of each line segment should scale with the recorded pen pressure
        :param tight: Whether to plot without margins
        :return:
        """
        if isinstance(color, Colormap):
            cmap = color
            color = None
        else:
            cmap = None

        points = np.column_stack((x, y))
        segments = []
        pressures = []
        cmap_array = []
        for i in range(len(points)-1):
            if s[i+1] < self.threshold:
                segments.append((points[i], points[i+1]))
                pressures.append((p[i] + p[i+1])/2)
                cmap_array.append(i / len(points))
        pressures = [press * pressure_scale for press in pressures]
        cmap_array = np.array(cmap_array)

        lines = LineCollection(segments, linewidths=pressures, colors=color, cmap=cmap, array=cmap_array)
        ax.add_collection(lines)

        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        if tight:
            x_mid = (x.min() + x.max()) / 2
            x_breadth = abs(x.min()) + abs(x.max())
            y_mid = (y.min() + y.max()) / 2
            y_breadth = abs(y.min()) + abs(y.max())
            breadth = max(x_breadth, y_breadth)
            ax.set_xlim(x_mid - breadth / 2 - 0.1, x_mid + breadth / 2 + 0.1)
            ax.set_ylim(y_mid - breadth / 2 - 0.1, y_mid + breadth / 2 + 0.1)
        else:
            ax.autoscale()
        lim = max(x.max(), abs(x.max()), y.max(), abs(y.max()))
        ax.set_aspect(1)

    def lineplot(self, params, rows, cols, path=None, show=False, row_labels=None, col_labels=None, transparent=True):
        """
        Helper function to call a tabled lineplot
        :param params:
        :param rows:
        :param cols:
        :param path:
        :param show:
        :param row_labels:
        :param col_labels:
        :param transparent:
        :return:
        """
        self.table(rows, cols, params, "line", path=path, show=show, row_labels=row_labels, col_labels=col_labels, transparent=transparent)

    def scatterplot(self, x, y, p, s, path=None, show=False):
        """
        Helper function to call a scatterplot for a single trajectory
        :param x:
        :param y:
        :param p:
        :param s:
        :param path:
        :param show:
        :return:
        """
        self.scatter_multiple([((x, y, p, s), {"color": plt.get_cmap("viridis"), "size": 100, "tight": True})], path=path, show=show)

    def scatter_multiple(self, params, path=None, show=False, transparent=False):
        """
        Helper function to call a scatterplot for a list of trajectories
        :param params:
        :param path:
        :param show:
        :return:
        """
        self.stack(params, "scatter", path=path, show=show, transparent=transparent)
