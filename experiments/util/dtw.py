import numpy as np


def timestep_distance_abs(timestep1, timestep2):
    """
    Absolute distance between two timesteps
    :param timestep1:
    :param timestep2:
    :return:
    """
    return np.abs(timestep1 - timestep2).sum()

def dtw_distance(trajectory_1, trajectory_2):
    """
    Calculates the dynamic time warping distance between two trajectories
    :param trajectory_1:
    :param trajectory_2:
    :return:
    """
    l1 = len(trajectory_1)
    l2 = len(trajectory_2)

    dtw = np.zeros((l1 + 1, l2+1))
    dtw[:, 0] = np.inf
    dtw[0, :] = np.inf
    dtw[0, 0] = 0.0

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            cost = timestep_distance_abs(trajectory_1[i - 1], trajectory_2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])

    return dtw[l1, l2]

def dtw_path(trajectory_1, trajectory_2):
    """
    Calculates the dynamic time warping distance and path between two trajectories
    :param trajectory_1:
    :param trajectory_2:
    :return:
    """
    l1 = len(trajectory_1)
    l2 = len(trajectory_2)

    dtw = np.zeros((l1 + 1, l2+1))
    dtw[:, 0] = np.inf
    dtw[0, :] = np.inf
    dtw[0, 0] = 0.0

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            cost = timestep_distance_abs(trajectory_1[i - 1], trajectory_2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])

    l = 1
    path = [(l1, l2)]

    n, m = l1, l2
    while not (n == 0 and m == 0):
        if n == 0:
            m -= 1
        elif m == 0:
            n -= 1
        else:
            idxs = np.array([(n - 1, m - 1), (n - 1, m), (n, m - 1)])
            n, m = idxs[np.argmin([dtw[n - 1, m - 1], dtw[n - 1, m], dtw[n, m - 1]])]
        l += 1
        path.append((n, m))

    return dtw[l1, l2], l, path
