import numpy as np


def mask_joint(frame, max_disturbance, no_mask_joint, m):
    """

    :param frame:
    :param max_disturbance:
    :param no_mask_joint:
    :param m:
    :return:
    """

    # Функция для добавления искажения в рендже [-max_disturbance; max_disturbance]
    def spatial_disturbance(xyz):
        disturbance = np.random.uniform(-max_disturbance, max_disturbance)
        return xyz + disturbance

    m = np.random.randint(1, m + 1)
    joints_idx_to_mask = np.random.choice(21, size=m, replace=False)
    list_of_values = []
    for key in frame:
        for point in frame[key]:
            list_of_values.append([point['x'], point['y'], point['z']])
    list_of_values = np.array(list_of_values)

    # Тут мы выбираем либо зануляем суставы, либо добавляем к ним искажение
    op_idx = np.random.binomial(1, p=0.5, size=m).reshape(-1, 1)
    for i, index in enumerate(joints_idx_to_mask):
        if op_idx[i] == 1:
            list_of_values[index] = spatial_disturbance(list_of_values[index])
        else:
            list_of_values[index] = spatial_disturbance(list_of_values[index]) \
                if not no_mask_joint else [0.0, 0.0, 0.0]

    return list_of_values


def mask_clip(frame_idx, seq, n_frames, K):
    """

    :param frame_idx:
    :param seq:
    :param n_frames:
    :param K:
    :return:
    """

    n_frames_to_mask = np.random.randint(2, K + 1)
    n_frames_to_mask_half = n_frames_to_mask // 2
    start_idx = frame_idx - n_frames_to_mask_half
    end_idx = frame_idx + n_frames_to_mask_half
    # Тут надо учесть что за рамки массива выйдет чиселки
    if start_idx < 0:
        diff = abs(start_idx)
        start_idx = 0
        end_idx += diff
    if end_idx > n_frames:
        diff = end_idx - n_frames
        end_idx = n_frames
        start_idx -= diff

    masked_frames_idxs = range(start_idx, end_idx)
    seq[masked_frames_idxs] = 0.0

    return seq, masked_frames_idxs


seq = [{'hand 1': [{'x': 0.298, 'y': 0.529, 'z': 0.0}, {'x': 0.354, 'y': 0.513, 'z': 0.005},
                   {'x': 0.389, 'y': 0.493, 'z': -0.004}, {'x': 0.422, 'y': 0.482, 'z': -0.017},
                   {'x': 0.452, 'y': 0.477, 'z': -0.03}, {'x': 0.324, 'y': 0.448, 'z': 0.008},
                   {'x': 0.39, 'y': 0.441, 'z': -0.014}, {'x': 0.422, 'y': 0.452, 'z': -0.035},
                   {'x': 0.441, 'y': 0.467, 'z': -0.046}, {'x': 0.311, 'y': 0.448, 'z': -0.007},
                   {'x': 0.388, 'y': 0.437, 'z': -0.032}, {'x': 0.422, 'y': 0.45, 'z': -0.048},
                   {'x': 0.443, 'y': 0.466, 'z': -0.055}, {'x': 0.306, 'y': 0.455, 'z': -0.024},
                   {'x': 0.377, 'y': 0.445, 'z': -0.042}, {'x': 0.412, 'y': 0.454, 'z': -0.049},
                   {'x': 0.434, 'y': 0.465, 'z': -0.049}, {'x': 0.311, 'y': 0.468, 'z': -0.041},
                   {'x': 0.36, 'y': 0.46, 'z': -0.052}, {'x': 0.39, 'y': 0.459, 'z': -0.054},
                   {'x': 0.413, 'y': 0.458, 'z': -0.053}]},
       {'hand 1': [{'x': 0.62, 'y': 0.88, 'z': -0.0}, {'x': 0.611, 'y': 0.844, 'z': -0.01},
                   {'x': 0.585, 'y': 0.817, 'z': -0.019}, {'x': 0.565, 'y': 0.792, 'z': -0.029},
                   {'x': 0.552, 'y': 0.774, 'z': -0.038}, {'x': 0.531, 'y': 0.849, 'z': -0.013},
                   {'x': 0.512, 'y': 0.846, 'z': -0.022}, {'x': 0.518, 'y': 0.847, 'z': -0.031},
                   {'x': 0.526, 'y': 0.849, 'z': -0.038}, {'x': 0.52, 'y': 0.875, 'z': -0.012},
                   {'x': 0.505, 'y': 0.879, 'z': -0.012}, {'x': 0.507, 'y': 0.881, 'z': -0.016},
                   {'x': 0.51, 'y': 0.881, 'z': -0.021}, {'x': 0.52, 'y': 0.899, 'z': -0.01},
                   {'x': 0.496, 'y': 0.905, 'z': -0.01}, {'x': 0.489, 'y': 0.909, 'z': -0.011},
                   {'x': 0.483, 'y': 0.912, 'z': -0.014}, {'x': 0.526, 'y': 0.917, 'z': -0.008},
                   {'x': 0.505, 'y': 0.924, 'z': -0.006}, {'x': 0.497, 'y': 0.93, 'z': 0.0},
                   {'x': 0.492, 'y': 0.934, 'z': 0.004}]},
       {'hand 1': [{'x': 0.62, 'y': 0.88, 'z': -0.0}, {'x': 0.611, 'y': 0.844, 'z': -0.01},
                   {'x': 0.585, 'y': 0.817, 'z': -0.019}, {'x': 0.565, 'y': 0.792, 'z': -0.029},
                   {'x': 0.552, 'y': 0.774, 'z': -0.038}, {'x': 0.531, 'y': 0.849, 'z': -0.013},
                   {'x': 0.512, 'y': 0.846, 'z': -0.022}, {'x': 0.518, 'y': 0.847, 'z': -0.031},
                   {'x': 0.526, 'y': 0.849, 'z': -0.038}, {'x': 0.52, 'y': 0.875, 'z': -0.012},
                   {'x': 0.505, 'y': 0.879, 'z': -0.012}, {'x': 0.507, 'y': 0.881, 'z': -0.016},
                   {'x': 0.51, 'y': 0.881, 'z': -0.021}, {'x': 0.52, 'y': 0.899, 'z': -0.01},
                   {'x': 0.496, 'y': 0.905, 'z': -0.01}, {'x': 0.489, 'y': 0.909, 'z': -0.011},
                   {'x': 0.483, 'y': 0.912, 'z': -0.014}, {'x': 0.526, 'y': 0.917, 'z': -0.008},
                   {'x': 0.505, 'y': 0.924, 'z': -0.006}, {'x': 0.497, 'y': 0.93, 'z': 0.0},
                   {'x': 0.492, 'y': 0.934, 'z': 0.004}]},
       {'hand 1': [{'x': 0.62, 'y': 0.88, 'z': -0.0}, {'x': 0.611, 'y': 0.844, 'z': -0.01},
                   {'x': 0.585, 'y': 0.817, 'z': -0.019}, {'x': 0.565, 'y': 0.792, 'z': -0.029},
                   {'x': 0.552, 'y': 0.774, 'z': -0.038}, {'x': 0.531, 'y': 0.849, 'z': -0.013},
                   {'x': 0.512, 'y': 0.846, 'z': -0.022}, {'x': 0.518, 'y': 0.847, 'z': -0.031},
                   {'x': 0.526, 'y': 0.849, 'z': -0.038}, {'x': 0.52, 'y': 0.875, 'z': -0.012},
                   {'x': 0.505, 'y': 0.879, 'z': -0.012}, {'x': 0.507, 'y': 0.881, 'z': -0.016},
                   {'x': 0.51, 'y': 0.881, 'z': -0.021}, {'x': 0.52, 'y': 0.899, 'z': -0.01},
                   {'x': 0.496, 'y': 0.905, 'z': -0.01}, {'x': 0.489, 'y': 0.909, 'z': -0.011},
                   {'x': 0.483, 'y': 0.912, 'z': -0.014}, {'x': 0.526, 'y': 0.917, 'z': -0.008},
                   {'x': 0.505, 'y': 0.924, 'z': -0.006}, {'x': 0.497, 'y': 0.93, 'z': 0.0},
                   {'x': 0.492, 'y': 0.934, 'z': 0.004}]}]

R = 0.4
m = 4
K = 8
max_disturbance = 0.25
no_mask_joint = False

sequence = seq.copy()
# n_frames = len(np.where(np.any(sequence != 0.0, axis=1))[0])
n_frames_to_mask = int(np.ceil(R * len(sequence)))
frames_to_mask = np.random.choice(len(sequence), size=n_frames_to_mask, replace=False)

clipped_masked_frames = []
for f in frames_to_mask:
    grabed_frame = sequence[f]
    mask_idx = np.random.choice(3)  # 0: joint, 1: frame, 2: clip
    if mask_idx == 0:
        grabed_frame = mask_joint(grabed_frame, max_disturbance, no_mask_joint, m)
        sequence[f] = grabed_frame
    elif mask_idx == 1:
        grabed_frame[:] = 0.
        sequence[f] = grabed_frame
    else:
        grabed_frame, masked_frames_idx = mask_clip(f, sequence, len(sequence), K)
        clipped_masked_frames.extend(masked_frames_idx)

# Компилируем список всех замаскированных кадров для использования в лоссе
masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))
print(masked_frames_idx)
