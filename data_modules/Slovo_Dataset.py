import os
import mediapipe as mp
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from utils import mask_transform


class SlovoDataset(Dataset):
    """
    TODO написать "документацию" к классу
    """
    def __init__(self, root_dir, path_to_annotations_file, transforms=None,):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.annotations = pd.read_csv(path_to_annotations_file, sep='\t')
        self.label_encoder = LabelEncoder()
        self.annotations['label'] = self.label_encoder.fit_transform(self.annotations['text'])
        self.mp_hands = mp.solutions.hands.Hands()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Функция в зависимости от заданного индекса вытаскивает инфу из csv файла annotations.csv
        Затем проходимся по каждому фрейму и размечаем руки, нормализуем метки и объединяем их в один массив
        По 21 метке на руку, СНАЧАЛА ИДУТ МЕТКИ ПРАВОЙ РУКИ, ЗАТЕМ ЛЕВОЙ
        :param idx:
        :return hand_landmarks, label:
        """
        csv_info = self.annotations.iloc[idx]
        attachment_id = csv_info['attachment_id']
        begin_frame = csv_info['begin']
        end_frame = csv_info['end']
        video_path = os.path.join(self.root_dir, f'{attachment_id}.mp4')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео {video_path}")

        hand_landmarks = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count < begin_frame:
                continue
            if frame_count > end_frame:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(image_rgb)

            left_hand_landmarks = None
            right_hand_marks = None

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if handedness.classification[0].label == 'Left':
                        left_hand_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    elif handedness.classification[0].label == 'Right':
                        right_hand_marks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            if left_hand_landmarks is not None and right_hand_marks is not None:
                hand_landmarks.append(right_hand_marks + left_hand_landmarks)
            elif left_hand_landmarks is not None:
                hand_landmarks.append([[0.0, 0.0, 0.0]] * 21 + left_hand_landmarks)
            elif right_hand_marks is not None:
                hand_landmarks.append(right_hand_marks + [[0.0, 0.0, 0.0]] * 21)

        cap.release()
        if not hand_landmarks:
            raise ValueError(f"Не было сделано меток для видео {attachment_id}")

        hand_landmarks = torch.tensor(np.array(hand_landmarks), dtype=torch.float32)
        hand_landmarks = normalize_landmarks(hand_landmarks)
        hand_landmarks_masked, masked_frames_idx = mask_transform(hand_landmarks, self.R,
                                                                  self.max_disturbance,
                                                                  self.no_mask_joint,
                                                                  self.K, self.m)
        label = torch.tensor(csv_info['label'], dtype=torch.long)

        if self.transforms:
            hand_landmarks_masked = self.transforms(hand_landmarks_masked)

        return hand_landmarks, hand_landmarks_masked, label, masked_frames_idx
