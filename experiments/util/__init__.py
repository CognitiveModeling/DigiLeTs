import numpy as np
import torch

def extract_vals(batched_values, i, sum=True):
    """Takes a batched value tensor and converts it to a tuple of numpy arrays,
    optionally summing the dx and dy values into positions x and y"""
    batched_values = batched_values.cpu()
    x, y, p, s = batched_values[:, i, 0], batched_values[:, i, 1], batched_values[:, i, 2], batched_values[:, i, 3]
    x, y, p, s = x.detach().numpy(), y.detach().numpy(), p.detach().numpy(), s.detach().numpy()
    if sum:
        x, y = np.cumsum(x), np.cumsum(y)
    return x, y, p, s


def onehot_to_label(input):
    """Converts a one-hot encoded label into the actual label"""
    potential = torch.where(input == 1)
    if len(potential[0]) > 0:
        return potential[0][0].item()
    else:
        return -1

def stretch_constant_interval(min, max, length, stretchmap):
    stretched = np.array([])
    x = min
    y = 0
    for i, segment in enumerate(stretchmap):
        s_vrange, s_lrange = segment
        s_min = x
        s_max = s_min + s_vrange * max
        if i == len(stretchmap) - 1:
            s_length = length - y
            stretched = np.append(stretched, np.linspace(s_min, s_max, s_length, endpoint=True))
        else:
            s_length = int(s_lrange * length) - 1
            stretched = np.append(stretched, np.linspace(s_min, s_max, s_length, endpoint=False))

        x = s_max
        y += s_length
    return stretched

class MultiSlice:
    """Class implementing the quick addition of multiple slices,
    specifically the element check and converting all slices to a list.
    e.g. is 12 in [1-9, 15-41, 99-112]."""
    def __init__(self, slices):
        self.slices = slices

    def __contains__(self, item):
        for (start, stop, step) in self.slices:
            # instant abort, slice encompasses everything
            if start is None and stop is None:
                return True
            # if slice encompasses something
            if stop is None or start < stop:
                # default assumption
                if start is None:
                    start = 0
                if step is None:
                    step = 1

                if item >= start and (stop is None or item < stop) and (item - start) % step == 0:
                    return True
        return False

    def __iter__(self):
        for start, stop, step in self.slices:
            for item in range(start, stop, step):
                yield item

    def __str__(self):
        str = "multislice(["
        for i, (start, stop, step) in enumerate(self.slices):
            str += f"({start}, {stop}, {step})"
            if i != len(self.slices) - 1:
                str += ","
        str += "])"
        return str

    def __repr__(self):
        str = "multislice(["
        for i, (start, stop, step) in enumerate(self.slices):
            str += f"({start}, {stop}, {step})"
            if i != len(self.slices) - 1:
                str += ","
        str += "])"
        return str


def multislice(slices):
    return MultiSlice(slices)


def openslice():
    return MultiSlice([(None, None, None)])