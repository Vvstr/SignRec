import numpy as np


def mask_transform(seq, R, max_disturbance, no_mask_joint, K, m):
    """
    Функция для приминения к последовательности маскировок:
    1) Clip masking - маскировка на уровне видео, учимся временной
    составляющей, что один кадр жеста зависит от других кадров жеста
    ВАЖНО ЧТОБЫ НЕСКОЛЬКО ЖЕСТОВ НЕ БЫЛО НА ОДНОМ ВИДЕО
    2) Joint masking - маскировка суставов, тут как с clip masking только
    учим модель пространственным признакам, то есть учитывать положение
    сустава по другим суставам
    :param seq: np.ndarray формой(N - количество кадров, H - высота, W - ширина, C - количество каналов)
    :param R: доля кадров, которые будут замаскированы
    :param max_disturbance:
    :param no_mask_joint: Список суставов, которые не подлежат маскировке, по сути для ключевых
    :param K: Количество кадров, которые будут ОДНОВРЕМЕННО замаскированы при clip masking
    :param m: Количество суставов для маскировки
    :return: tuple np.ndarray Замаскированная последовательность
                   np.ndarray Индексы замаскированных кадров
    """
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

    return sequence, masked_frames_idx


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
        disturbances = np.random.uniform(-max_disturbance, max_disturbance, (xyz.shape[0], 3))
        return xyz + disturbances

    m = np.random.randint(1, m + 1)
    joints_idx_to_mask = np.random.choice(21, size=m, replace=False)
    # Тут мы выбираем либо зануляем суставы, либо добавляем к ним искажение
    op_idx = np.random.binomial(1, p=0.5, size=m).reshape(-1, 1)
    frame[joints_idx_to_mask] = np.where(
        op_idx,
        spatial_disturbance(frame[joints_idx_to_mask]),
        spatial_disturbance(frame[joints_idx_to_mask]) if no_mask_joint else 0.0
    )

    return frame


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
