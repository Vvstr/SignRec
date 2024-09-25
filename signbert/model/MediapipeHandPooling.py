import torch
import torch.nn as nn

from IPython import embed; from sys import exit


class MediapipeHandPooling(nn.Module):
    """
    Pytorch модуль для объединения кипоинтов рук
    MediaPipe для рук имеет 21 точку, мы их объединяем, чтобы могли с ними работать
    Attributes:
        PALM_IDXS, THUMB_IDXS, INDEX_IDX, MIDDLE_IDX, RING_IDX, PINKY_IDX: Кортежи, которые
        содержат индексы кипоинтов для каждой части руки
    """
    PALM_IDXS = (0, 1, 5, 9, 13, 17)
    THUMB_IDX = (2, 3, 4)
    INDEX_IDXS = (6, 7, 8)
    MIDDLE_IDXS = (10, 11, 12)
    RING_IDXS = (14, 15, 16)
    PINKY_IDXS = (18, 19, 20)

    def __init__(self, last=False):
        """
        Инициализация пулинг метода
        last(bool): Если True, то это значит что пулинг метод был уже сделан, поэтому
        новым пулинг будет применятся только к последнему измерению. Default = False
        """
        super().__init__()
        self.last = last

    def forward(self, x):
        """
        Собственно применение пулинга к кипоинтам рук
        Params:
        x(Tensor): Тезнор, содержащий кипоинты рук
        Returns:
        Tensor: Тензор после пулинга
        """
        # Если пулинг был сделан, нужно проверить имеем ли мы только 6 измерений,
        # мб содержаться кипоинты ещё чего-то, что нам не нужно
        if self.last:
            assert x.shape[3] == 6
            return torch.amax(x, 3, keepdim=True)
        else:
            assert x.shape[3] == 21
            return torch.cat((
                torch.amax(x[:, :, :, MediapipeHandPooling.PALM_IDXS], 3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.THUMB_IDX], dim=3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.INDEX_IDXS], dim=3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.MIDDLE_IDXS], dim=3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.RING_IDXS], dim=3, keepdim=True),
                torch.amax(x[:, :, :, MediapipeHandPooling.PINKY_IDXS], dim=3, keepdim=True)
            ), dim=3)
