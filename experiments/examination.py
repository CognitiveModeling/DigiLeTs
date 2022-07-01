__author__ = "Julius Wührer"

"""
Sample examination / utility functions
"""
import torch
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

from models.encoding import Encoding

import dataset.trajectories_data as traj_data
from dataset.trajectories_data import TrajectoriesDataset, ToTensor, num_to_char
from util.dtw import dtw_distance, dtw_path

from util.plotting import Plotter
from util import multislice, extract_vals


def load(model, path):
    """
    Loads the state dict from the contained path into the supplied model
    :param model:
    :param path:
    :return:
    """
    dic = torch.load(path)
    model.load_state_dict(dic['model_state_dict'])


########################################################################################################################
# DTW and Imitation Measure                                                                                            #
########################################################################################################################

def dtw_total(parts, chars, insts, pressure, device, model=None, name="original"):
    """
    Calculates the DTW matrix in it's entirety, a numpy array of shape (len(parts), len(parts), len(chars), len(insts), len(insts))
    :param parts: The iterable (list, range etc.) of participant ids
    :param chars: Iterable of character ids
    :param insts: Iterable of instance ids
    :param pressure: Whether to include pressure for dtw calculation
    :param device: Device tensors should be on
    :param model: Model to generate trajectories to calculate dtw distances with dataset trajectories with. If none,
    calculate dataset to dataset DTW distance (can save calculation since matrix then symmetric on main diagonal)
    :param name: Name to save matrix under as a npy file
    :return:
    """
    trans_traj = transforms.Compose([ToTensor(), traj_data.ToOnehot(62, 77)])
    data_traj = TrajectoriesDataset("./data.pickle", transform=trans_traj, load_pretransformed=True)
    dataset_to_model = model is None

    dtw_matrix = np.zeros((len(parts), len(parts), len(chars), len(insts), len(insts)))
    lengths = np.zeros((len(parts), len(parts), len(chars), len(insts), len(insts), 3))
    for i, part1 in enumerate(parts):
        print(f"Row: Participant {part1}")
        parts_partial = enumerate(parts)
        if not dataset_to_model:
            parts_partial = [*enumerate(parts)][i:]
        for j, part2 in parts_partial:
            print(f"\tCol: Participant {part2}")
            for k, char in enumerate(chars):
                # print(f"\t\tChar {char}")
                for l, inst2 in enumerate(insts):
                    output, input_char, input_part, length, _ = data_traj.get_pci(part1, char, inst2)
                    input_char, input_part = input_char.to(device), input_part.to(device)
                    if dataset_to_model:
                        output = model(input_char, input_part, length)
                    x, y, p, s = extract_vals(output, 0)
                    stack = np.column_stack((x, y))

                    # only if net actually predicted pressure, crash otherwise
                    if pressure:
                        stack = np.column_stack((stack, p))

                    to_compare = data_traj.get_pci(part2, char, inst2)
                    to_compare = [(m, data_traj.get_pci(part2, char, inst)) for (m, inst) in enumerate(insts)]
                    for m, compare in to_compare:
                        target, _, _, _, _ = compare
                        tx, ty, tp, ts = extract_vals(target, 0)
                        tstack = np.column_stack((tx, ty))
                        if pressure:
                            tstack = np.column_stack((tstack, tp))

                        dist, length, _ = dtw_path(stack, tstack)
                        dtw_matrix[i, j, k, l, m] = dist
                        lengths[i, j, k, l, m] = length, tx.shape[0], x.shape[0]

                        if not dataset_to_model:
                            dtw_matrix[j, i, k, l, m] = dist
                            lengths[j, i, k, l, m] = length, tx.shape[0], x.shape[0]

    pstring = f"{min(parts)}-{max(parts)}" \
        if isinstance(parts, range) \
        else "".join(parts)
    cstring = f"{num_to_char(min(chars))}-{num_to_char(max(chars))}" \
        if isinstance(chars, range) \
        else "".join([num_to_char(c) for c in chars])
    istring = f"{min(insts)}-{max(insts)}" \
        if isinstance(insts, range) \
        else "".join(insts)
    np.save(
        f"examination/dtw/total/matrix_{name}_"
        f"{pstring}_{cstring}_{istring}",
        dtw_matrix)

    np.save(
        f"examination/dtw/total/lengths_{name}_"
        f"{pstring}_{cstring}_{istring}",
        lengths)


def dtw_matrix(dtw_total):
    """
    Retrieving the averaged DTW value for each participant compared to each participant
    :param dtw_total: Total dtw matrix of shape (
    :return:
    """
    return np.mean(dtw_total, axis=(2, 3, 4))

def imitation_measure(dtw_original, dtw_generated):
    errors = (dtw_generated - dtw_original) ** 2
    imitation_measure = errors.mean()
    # calculating self similarity error (mse of main diagonals)
    diagonal_imitation_measure = np.diag(errors, k=0).mean()
    # calculating foreign similarity error (mse of everything except main diagonals)
    np.fill_diagonal(errors, 0)
    ul_triangle_imitation_measure = errors[errors != 0].mean()
    return imitation_measure, diagonal_imitation_measure, ul_triangle_imitation_measure


def measure_dtw_total():
    device = torch.device("cuda")
    model = Encoding(character_size=62, participant_size=77, hidden_size=100, embedding_size=50, num_layers=1,
                          hidden_bias=False, dropout=0, output_size=4, output_bias=False, inf=True, infer_target="both")
    model.to(device)
    # load(model, "runs/extomni/oneshot/models/Oneshot_epoch999.pth")
    dtw_total(parts=range(0, 77),
              chars=range(10, 36),
              insts=range(0, 5),
              pressure=False,
              device=device,
              model=model,
              name="encoding")

########################################################################################################################
# Plotting                                                                                                             #
########################################################################################################################

def plot_by_participant(model=None):
    trans_traj = transforms.Compose([ToTensor(), traj_data.ToOnehot(62, 77)])
    data_traj = TrajectoriesDataset("./data.pickle", transform=trans_traj, load_pretransformed=True)

    plotter = Plotter(0.9, "pdf")

    for p in range(77):
        params = []
        for c in range(62):
            for i in range(5):
                target, input, part, length, _ = data_traj.get_pci(p, c, i)
                tx, ty, tpr, ts = extract_vals(target, 0)
                params.append(((tx, ty, tpr, ts), {"color": plt.get_cmap("viridis"), "pressure_scale": 3}))
                if model is not None:
                    output = model(input, part, length)
                    x, y, pr, s = extract_vals(output, 0)
                    params.append(((x, y, pr, s), {"color": plt.get_cmap("viridis"), "pressure_scale": 3}))
        plotter.lineplot(params, 62, 10 if model is not None else 5, path=f"./examination/visualize/participant-{p}")


def plot():
    device = torch.device("cpu")
    model = Encoding(character_size=62, participant_size=77, hidden_size=100, embedding_size=50, num_layers=1,
                     hidden_bias=False, dropout=0, output_size=2, output_bias=False, inf=True, infer_target="both")
    model.to(device)
    load(model, "runs/extomni/oneshot/models/Oneshot_epoch999.pth")

    plot_by_participant(model)


if __name__ == '__main__':
    plot()
