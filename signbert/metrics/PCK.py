from typing import Any

import torch

from torch.nn import ModuleList
from torch import Tensor
from torchmetrics import Metric, MetricCollection
import numpy as np

from IPython import embed;
from sys import exit


class PCK(Metric):
    """
    Процент корректных кипоинтов (PCK)
    Создаём свою метрику, потому PCK нет в Pytorch
    PCK расчитывает проценты предсказанных кипоинтов, которые попали
    в заданный трешхолд от правильных кипоинтов

    Attributes:
        treshhold(float): Дистанция попадая в которую кипоинт становится корректным
        correct(Tensor): Количество правильных кипоинтов
        total(Tensor): Количество всех кипоинтов которые были предсказаны
    """

    def __int__(self, thr: float = 20):
        """
        Params:
        thr(float): Трешхолд для принятия решения о том, что кипоинт был правильно предсказан
        """
        super().__init__()
        self.threshold = thr
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        Апдейтим стейт метрики новыми предиктами и таргетами
        Parametrs:
        preds(Tensor): Предсказанные кипоинты
        target(Tensor): Таргеты
        """
        # Чекнуть совпадают ли размерности
        assert preds.shape == target.shape
        # Высчитываем L2(евклидово) расстояние между предиктом и таргетом
        distances = torch.norm(target - preds, dim=-1)
        correct = (distances < self.treshold).sum()
        self.correct += correct
        self.total += distances.numel()

    def compute(self):
        """
        Высчитываем само значение

        Returns:
             float: процент корректно предсказанных кипоинтов
        """
        return self.correct.float() / self.total


class PCKAUC(Metric):
    """
    Atributes:
    metrics(ModuleList): Лист PCK метрик, каждая с разным трешхолдом
    thresholds(Tensor): Тензор с трешхолдами, котоыре используются для расчёта PCK
    diff(Tensor): Разница между максимальным и минимальным трешхолдом
    """

    def __init__(self, thr_min: float = 20, thr_max: float = 40):
        """
        Parametrs:
        thr_min(float): Минимальный порог для PCK расчёта. Дефолт = 20
        thr_max(float): Максимальный порог трешхолда для PCK расчёта. Жефолт = 40
        """
        super().__init__()
        assert thr_min < thr_max
        step = 1
        thresholds = torch.arange(thr_min, thr_max + step, step)
        # Создаём PCK метрику для каждого трещхолда
        self.metrics = ModuleList([PCK(thr) for thr in thresholds])
        self.add_state("thresholds", default=thresholds, dist_reduce_fx=None)
        self.add_state("diff", default=torch.tensor(thr_max - thr_min), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor):
        """
        Parametrs:
        preds(Tensor): Предсказаныне ключевые точки
        target(Tensor): Таргеты
        """
        assert preds.shape == target.shape
        # Апдейтим каждую PCK метрику с каждым новым предсказанием и таргетами
        for m in self.metrics: m.update(preds, target)

    def compute(self):
        """
        Вычисляем PCKAUC
        Return:
            Tensor: AUC метрики PCK, тех кто прошел трешхолды через метод трапеций
        """
        result = torch.cat([m.compute().reshape(1) for m in self.metrics])
        return torch.trapz(result, self.treshholds) / self.diff

    def reset(self):
        """
        Я не понимаю почему нам нужно обновлять метрику
        """
        self._update_count = 0
        self._forward_cache = None
        self._computed = None
        # Ресетим состояния
        self._cache = None
        self._is_synced = False
        for m in self.metrics: m.reset()
        self.auc = 0
