import torch
import torch.nn as nn

from IPython import embed


class ArmExtractor(nn.Module):
    """
    Модуль для извлечения и обработки ключевых точек рук с использованием STGCN
    Я ничего не понял, но говорят, что GCN работают лучше RNN и CNN
    для извлечения признаков
    Атрибуты:
    stgcn: пространственно-временной граф
    """

    def __init__(
            self,
            in_channels,
            hid_dim,
            dropout
    ):
        """
        Параметры:
        in_channels (int): Количетсво входных канналов(фичей).
        hid_dim:  Размреность скрытого слоя в моделе stgcn.
        dropout: Дропаут регуляризация в stgcn.
        """
        super().__init__()
        self.stgcn = STCGN(
            in_channels=in_channels,
            hid_dim=hid_dim,
            graph_args={'layout': 'mmpose arms'},
            dropout=dropout,
            egde_importance_weighting=False,
        )

    def forward(self, x):
        #Считаем длинну последовательности(не включая нулевые)
        lens = (x != 0.0).all(-1).all(-1).sum(1)
        #Переставляем значения в последовательности и решейпим тензор для STGNC
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        # Загоняем x в TGCN
        x = self.stgcn(x, lens)
        #Забираем индексы левой и правой руки
        rarm = x[:, :, :, (1, 3, 5)]
        larm = x[:, :, :, (0, 2, 4)]
        # Применяем maxpooling к точкам рук
        rarm = torch.amax(rarm, dim=3, keepdim=True)
        larm = torch.amax(rarm, dim=3, keepdim=True)

        return (rarm, larm)

