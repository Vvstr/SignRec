import random

import numpy as np


def mask_transform_indentity(seq, R, max_disturbance, no_mask_joint, K, m):
    """
    Эта функция применяет рандомно одну из нескольких макскировок на указанную последовательность фреймов
    Типы максировки: joint masking, frame masking, clip masking и без изменений

    Params:
    seq (numpy.ndarray): Последовательность фреймов, которые нужно замаскировать
    R(int): Заданная пропорция замаскированных фреймов к всем фреймам

    Returns:
    tuple:
        - numpy.ndarray: Трансформированная последовательность с применённой маскировкой
        - numpy,ndarray: Индексы фреймов, которые были замаскированы
    """
    toret = seq.copy()
    # Сначала считаем количество фреймов, которые на замаскированы,
    # а потом считаем количество фреймов которые нужно замаскировать
    n_frames = (toret != 0.0).all(1, 2).sum()
    n_frames_to_mask = int(np.ceil(R * n_frames))
    # Рандомно выбираем индексы фреймов для маскировки
    frames_to_mask = np.random.choice(n_frames, size=n_frames_to_mask, replace=False)
    clipped_masked_frames = []
    # Начинаем максировку в зависимости от предпочтения по типу маскировки(тут рандомно)
    for f in frames_to_mask:
        curr_frame = toret[f]
        op_idx = np.random.choice(4)

        # Если выбран joint mask
        if op_idx == 0:
            curr_frame = mask_joint(curr_frame, max_disturbance, no_mask_joint, m)
            toret[f] = curr_frame
        # Если выбран frame masking
        elif op_idx == 1:
            curr_frame[:] = 0.
            toret[f] = curr_frame
        # Если выбран clip masking
        elif op_idx == 2:
            curr_frame, masked_frames_idx = mask_clip(f, toret, n_frames, K)
            clipped_masked_frames.extend(masked_frames_idx)
        # Ничего
        else:
            pass
    # Собираем список всех замаскированных фреймов для использования в расчёте лосса
    masked_frames_idx = np.unique(np.concatenate((frames_to_mask, clipped_masked_frames)))

    return toret, masked_frames_idx


def mask_clip(frame_idx, seq, n_frames, K):
    """
     mask_clip предполагает установку непрерывного подмножество фреймов в 0. Длина клипа для маскировки берётся рандомно,
     центрировано вокруг заданного индекса

     Params:
     frame_idx(int): Индекс фрейма, которые будет опорным
     seq (numpy.ndarray): Последовательность фреймов к которой маскировка будет применена
     n_frames(int): Общее количество фреймов в последовательности

     Returns:
     tuple:
        -numpy.ndarray: Последовательность с применённым clip masking
        -list: Индексы фреймов, которые были замаскированы
    """
    # Рандомно выбираем количество фреймов для маскировки, с максимум в K фреймов
    n_frames_to_mask = np.random.randint(2, K+1)
    n_frames_to_mask_half = n_frames_to_mask // 2
    start_idx = frame_idx - n_frames_to_mask_half
    end_idx = frame_idx + (n_frames_to_mask - n_frames_to_mask_half)
    # Изменяем начальные и конечные индексы если они выходят за рамки
    if start_idx < 0:
        diff = abs(start_idx)
        start_idx = 0
        end_idx += diff
    if end_idx > n_frames:
        diff = end_idx - n_frames
        end_idx = n_frames
        start_idx -= diff
    # Генерируем список индексов для фреймов, которые нужно замаскировать
    masked_frames_idx = list(range(start_idx, end_idx))
    seq[masked_frames_idx] = 0.0

    return seq, masked_frames_idx


def mask_joint(frame, max_disturbance, no_mask_joint, m):
    """
    Эта функция выбирает случаные количество joints в заданном фрейме и применяет либо zero-masking
    либо spatial disturbance к этим joints. Zero-masking выставляет координаты joint в 0,
    spatial disturbance добавляет рандомное смещение к координатам

    Parameters:
    frame (numpy.ndarray): The frame (array of joint coordinates) to be masked.

    Returns:
    numpy.ndarray: The frame with masking applied to specific joints.
    """

    # Define a function for spatial disturbance
    def spatial_disturbance(xy):
        # Add a random disturbance within the range [-max_disturbance, max_disturbance]
        return xy + [np.random.uniform(-max_disturbance, max_disturbance),
                     np.random.uniform(-max_disturbance, max_disturbance)]

    # Randomly decide the number of joints to mask, with a maximum of 'm'
    m = np.random.randint(1, m + 1)
    # Randomly select joint indices to mask
    joint_idxs_to_mask = np.random.choice(21, size=m, replace=False)
    # Randomly decide the operation to be applied: zero-masking or spatial disturbance
    op_idx = np.random.binomial(1, p=0.5, size=m).reshape(-1, 1)
    # Apply the chosen masking operation to the selected joints
    frame[joint_idxs_to_mask] = np.where(
        op_idx,
        spatial_disturbance(frame[joint_idxs_to_mask]),
        spatial_disturbance(frame[joint_idxs_to_mask]) if no_mask_joint else 0.0
    )

    return frame
