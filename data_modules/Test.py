import os
import cv2
import torch
from torch.utils.data import DataLoader
from Slovo_Dataset import SlovoDataset, combined_dataset_collate_fn


root_dir = 'Slovo/train'
annotations_file = 'Slovo/annotations.csv'
batch_size = 1

# Создание экземпляра датасета
dataset = SlovoDataset(root_dir=root_dir, path_to_annotations_file=annotations_file)

# Создание DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=combined_dataset_collate_fn)

# Тестирование датасета
for idx, (hand_landmarks, hand_landmarks_masked, labels, masked_frame_idxs) in enumerate(data_loader):
    print(f"Batch {idx + 1}:")
    print(f"Original Hand Landmarks Shape: {hand_landmarks.shape}")
    print(f"Masked Hand Landmarks Shape: {hand_landmarks_masked.shape}")
    print(f"Labels: {labels}")
    print(f"Masked Frame Indexes: {masked_frame_idxs}")

    # Для визуализации ключевых точек
    # Преобразуем hand_landmarks в numpy для удобства
    hand_landmarks_np = hand_landmarks.detach().numpy()

    # Отображение ключевых точек на первом кадре
    for frame in hand_landmarks_np[0]:  # Берем первый элемент из батча
        frame = frame.reshape(-1, 3)  # Убедитесь, что форма правильная (N, 3)
        for landmark in frame:
            # Отображаем ключевые точки на черном фоне
            cv2.circle(frame, (int(landmark[0] * 640), int(landmark[1] * 480)), 5, (0, 255, 0), -1)

    cv2.imshow('Hand Landmarks', frame)
    cv2.waitKey(0)  # Ждем нажатия клавиши
    cv2.destroyAllWindows()

    # Прерываем после первого батча для тестирования
    break
