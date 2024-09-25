import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Класс для добавления кодинга пространства к имеющейся последовательности
    Attributes:
        dropout: nn.Dropout
        pe(tensor): Тензор позиционного кодирования
    """
    def __int__(self, d_model, dropout=0.1, max_len=5000):
        """
        dropout: Дропаут
        d_model(int): Пространственные эмбеддинги
        max_len(int): Максимальная длинна последовательности, которую мы будем обрабатывать
        """
        super(PoisionalEncoding, self).__int__()
        self.Dropout = nn.Dropout(p=dropout)
        # Создаем матрицу с позиционными кодировками с синусоидальной функцией
        pe = torch.zeros(max_len, d_model)
        position = torch.arrange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Тут мы создаём вектор скейлеров, которые используются в трансформерах
        # В трансформерах испольлзутеся синусоидальное кодирование(возможно как один из вариантов)
        # И тип в зависимости от позиции и размерности признаков
        # Мы получаем уникальное представление с помощью синуса
        # Что помогает отражать относительные позиции в последовательности(Like)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Также синусы и косинусы также нам дают позиционное кодирование
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # unsqueeze для того, чтобы добавить новое измерение, в виде количества каналов как я понял
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
