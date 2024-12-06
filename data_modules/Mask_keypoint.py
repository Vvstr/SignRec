import json

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import mask_transform
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MaskKeyPointDataset(Dataset):
    def __init__(self, json_fpath, path_to_annotations_file, R=0.4, m=4, K=8, max_disturbance=0.25,
                 no_mask_joint=False):
        super().__init__()
        with open(json_fpath, 'r') as f:
            self.data = json.load(f)

        self.seq_array = []
        self.videos = list(self.data.keys())
        self.R = R  # 0.4
        self.m = m  # -
        self.K = K  # 8
        self.max_disturbance = max_disturbance
        self.no_mask_joint = no_mask_joint
        self.annotaions_file = pd.read_csv(path_to_annotations_file, sep='\t')
        self.label_encoder = LabelEncoder()
        self.annotaions_file['label'] = self.label_encoder.fit_transform(self.annotaions_file['text'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_idx = self.videos[idx]
        seq = self.data[seq_idx]
        score = self.annotaions_file.iloc[idx]['label']
        seq_masked, masked_frames_idx = mask_transform(seq, self.R, self.max_disturbance,
                                                       self.no_mask_joint, self.K, self.m)

        return seq_idx, seq, seq_masked, score, masked_frames_idx


def mask_keypoint_dataset_collate_fn(batch):
    idxs = []
    seqs = []
    seqs_masked = []
    scores = []
    masked_frame_idxs = []
    n_masked_frames_idxs = [len(b[4]) for b in batch]
    max_masked_frames = max(n_masked_frames_idxs)
    pad_value = max_masked_frames - np.array(n_masked_frames_idxs)
    for i in range(len(batch)):
        idx, seq, seq_masked, score, frame_idxs = batch[i]
        idxs.append(idx)
        seqs.append(seq)
        seqs_masked.append(seq_masked)
        scores.append(score)
        masked_frame_idxs.append(np.pad(frame_idxs, (0, pad_value[i]), mode='constant', constant_values=-1.))
    idxs = np.array(idxs)
    seqs = np.stack(seqs)
    seqs_masked = np.stack(seqs_masked)
    scores = np.stack(scores)
    masked_frame_idxs = np.stack(masked_frame_idxs)
    idxs = torch.tensor(idxs, dtype=torch.int32)
    seqs = torch.tensor(seqs, dtype=torch.float32)
    seqs_masked = torch.tensor(seqs_masked, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    masked_frame_idxs = torch.tensor(masked_frame_idxs, dtype=torch.int64)

    return idxs, seqs, seqs_masked, scores, masked_frame_idxs


if __name__ == '__main__':
    import time

    # Замените путь на ваш тестовый JSON-файл
    json_fpath = 'Slovo/slovo_mediapipe.json'
    csv_path = 'Slovo/annotations.csv'

    # Создайте экземпляр вашего датасета
    dataset = MaskKeyPointDataset(
        json_fpath=json_fpath,
        path_to_annotations_file=csv_path
    )

    # Профилируем DataLoader
    num_samples = 10
    batch_size = 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mask_keypoint_dataset_collate_fn
    )

    elapsed_record = []
    for _ in range(num_samples):
        start_time = time.time()
        for batch in dataloader:
            pass
        elapsed = time.time() - start_time
        elapsed_record.append(elapsed)
        print(f"Time taken for {len(dataset)} samples: {elapsed:.4f} seconds")

    # Рассчитайте среднюю продолжительность
    average_duration = np.mean(elapsed_record)
    print(f"Average time per __getitem__: {average_duration:.4f} seconds")
