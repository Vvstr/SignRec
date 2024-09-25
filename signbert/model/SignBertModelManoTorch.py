import os
from typing import Any

import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from signbert.utils import my_import
from signbert.metrics.PCK import PCK, PCKAUC
from signbert.model.PositionalEncoding import PositionalEncoding
from manotorch.manotorch.manolayer import ManoLayer, MANOOutput
from IPython import embed;from sys import exit


class SignBertModel(pl.LightningModule):
    """
    Pytorch Lightning модуль имплементирующий модель SignBert для распознавания жестов
    Здесь используется извлечение жестовых признаков, позиционный енкодинг, пространственно-темпоральная обработка,
    энкодеры трансформер блока, и MANO слои для обработки и интерпретации жестов ЖЯ.

    Attributes:
    Разные конфигурационные параметры такие как: in_channels, num_hid, num_heads и т.д.
    ge(GestureExtraction): Модуль для извлечения жестовых признаков
    pe(PositionalEncoding): Модуль для позиционного кодирования
    stpe(SpatialTemporal): Модуль для пространственно-темпоральной обработки кипоинтов
    te(TransformerEncoder): Модуль для обработки последовательностей с помощью transformer
    pg(Linear): Линейный слой для предикта
    rhand_hd, lhand_hd(ManoLayer): MANO слои для детальной оценки поз рук
    PCK и PCKAUC метрики для тренировки и валидации
    """
    def __init__(
            self,
            in_channels,
            num_hid,
            num_heads,
            tformer_n_layers,
            tformer_dropout,
            eps,
            lmbd,
            weight_beta,
            weight_delta,
            lr,
            hand_cluster,
            n_pca_components,
            gesture_extractor_cls,
            gesture_extractor_args,
            total_steps=None,
            normalize_inputs=False,
            use_pca=True,
            flat_hand=False,
            weight_decay=0.01,
            use_onecycle_lr=False,
            pct_start=None,
            *args,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_hid = num_hid
        self.num_heads = num_heads
        self.tformer_n_layers = tformer_n_layers
        self.tformer_dropout = tformer_dropout
        self.eps = eps
        self.lmbd = lmbd
        self.weight_beta = weight_beta
        self.weight_delta = weight_delta
        self.total_steps = total_steps
        self.lr = lr
        self.hand_cluster = hand_cluster
        self.n_pca_components = n_pca_components
        self.gesture_extractor_args = gesture_extractor_args
        self.normalize_inputs = normalize_inputs
        self.use_pca = use_pca
        self.flat_hand = flat_hand
        self.weight_decay = weight_decay
        self.use_onecycle_lr = use_onecycle_lr
        self.pct_start = pct_start
        self.gesture_extractor_cls = my_import(gesture_extractor_cls)
        # Переменная для контроля количества входных каналов динамически, основываясь на доступности кластеринга
        num_hid_mult = 1 if hand_cluster else 21
        # Инициализация разных компонентов модели
        self.ge = self.gesture_extractor_cls(***gesture_extractor_args)
        el = torch.nn.TransformerEncoderLayer(d_model=num_hid*num_hid_mult, nhead=num_heads, batch_first=True, dropout=tformer_dropout)
        self.pe = PositionalEncoding(
            d_model=num_hid*num_hid_mult,
            dropout=0.1,
            max_len=2000
        )
        self.te = torch.nn.TransformerEncoder(el, num_layers=tformer_n_layers)
        self.pg = torch.nn.Linear(
            in_features=num_hid*num_hid_mult,
            out_features=(
                n_pca_components + 3 + #theta + 3D
                10 + #beta
                9 + # матрица поворота 3х3
                2 + # вектор переноса
                1
            )
        )
        # Доинициализируем оставшиеся компоненты модели
        mano_assets_root = os.path.split(__file__)[0]
        mano_assets_root = os.path.join(mano_assets_root, "thirdparty", "mano_assets")
        assert os.path.isdir(mano_assets_root), "Надо скачать MANO файлы"
        self.hd = ManoLayer(
            center_idx = 0,
            flat_hand_mean=flat_hand,
            mano_assets_root=mano_assets_root,
            use_pca=use_pca,
            ncomps=n_pca_components,
        )
        # PCK и PCKAUC метрики для тренировки и валидации
        self.train_pck_20 = PCK(thr=20)
        self.train_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        self.val_pck_20 = PCK(thr=20)
        self.val_pck_auc_20_40 = PCKAUC(thr_min=20, thr_max=40)
        # Холдеры
        self.train_step_losses = []
        self.val_step_losses = []

    def forward(self, x):
        # Достаём токены рук с помощью модуля GestureExtraction
        x = self.ge(x)
        # Убираем последнее измерение M и пермутим к виду (N - количество объектов, C - количество каналов,
        # T - временные признаки, V - количество точек)
        # Использовать contigous после перестановки - хорошая практика т.к.
        # после перестановки данные могут не находится последовательно, что как я понял замедляет из обработку
        x = x.squeeze(-1).permute(0, 2, 1, 3).contiguous()
        N, C, T, V = x.shape
        x = x.view(N, T, C*V)
        # Применяем позиционный энкодинг
        x = self.pe(x)
        # Прогоняем данные через трансформер энкодер
        x = self.te(x)
        # Предсказываем параметры рук и камеры -> прогоняем через Linear
        params = self.pg(x)
        # Извлекаем параметры рук, оффсет - написал гений, не я
        offset = self.n_pca_components + 3
        pose_coeffs = params[...,:offset]
        betas = params[...,offset:offset+10]
        offset += 10
        # Извлекаем параметры камеры
        R = params[...,offset:offset+9]
        R = R.view(N, T, 3, 3)
        offset += 9
        O = params[...,offset:offset+2]
        offset+=2
        S = params[...,offset:offset+1]
        # Решейпим параметры рук для обработки
        pose_coeffs = pose_coeffs.view(N*T, -1)
        betas = betas.view(N*T, -1)
        # Тут начинается приминение MANO модели для обработки 3D соединений и вершин(perevodchik) ->
        # Получаем 3D суставы и вершины
        mano_output: MANOOutput = self.hd(pose_coeffs, betas)
        # Извлекаем и решейпим MANO выходы
        vertices = mano_output.verts
        joints_3d = mano_output.joints
        pose_coeffs = pose_coeffs.view(N, T, -1)
        betas = betas.view(N, T, -1)
        vertices = vertices.view(N, T, 778, 3).detach().cpu()
        center_joint = mano_output.center_joint.detach().cpu()
        joints_3d = joints_3d.view(N, T, 21, 3)
        # Здесь надо применить ортографической проекцию к 3D, чтобы получить 2D
        # Применяем вращение к 3D суставам
        x = torch.matmul(R, joints_3d.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = x[...,:2]
        x *= S.unsqueeze(-1)
        x += O.unsqueeze(2)

        # На выходе получаем 2D координаты тысячи суставов, данные по камере, бетки и т.д.
        return x, pose_coeffs, betas, vertices, R, S, O, center_joint, joints_3d


    def training_step(self, batch):
        # Извлекаем дату из батчей
        _, x_or, x_masked, scores, masked_frames_idx = batch
        # Forward pass
        (logits, theta, beta, _, _, _, _, _, _) = self(x_masked)
        # Лосс применяется только на замаскированные данные, мы же BERT юзаем
        valid_idx = torch.where(masked_frames_idx != -1.)
        logits = logits[valid_idx]
        x_or = x_or[valid_idx]
        scores = scores[valid_idx]
        # Вычисляем потери реконструкции(предсказания)(LRec) и рег лосс(LReg)
        # Вычисляем L1 норму тут ес чо
        lrec = torch.norm(logits[scores > self.eps] - x_or[scores > self.eps], p=1, dim=1).sum()
        beta_t_minus_one = torch.roll(beta, shifts=1, dims=1)
        beta_t_minus_one[:, 0] = 0
        # Вычисляем изменение между текущим beta и предыдущим, умножая на коэффициент регуляризации
        lreg = torch.norm(theta, p=2) + self.weight_beta * torch.norm(beta, p=2) + \
            self.weight_delta * torch.norm(beta - beta_t_minus_one, p=2)
        # Комбинируем оба лосса
        loss = lrec + (self.lmbd * lreg)
        # Аппендим лосс шага, детач нужен чтобы "открепить" данные от общего пула
        # И сделать так, чтобы они не влияли на backprop
        self.train_step_losses.append(loss.detach().cpu())
        # Проверка на нормализованные данные
        if self.normalize_inputs:
            if not hasattr(self, 'means') or not hasattr(self, 'stds'):
                self.means = torch.from_numpy(np.load(self.trainer.datamodule.MEANS_NPY_FPATH)).to(self.device)
                self.stds = torch.from_numpy(np.load(self.trainer.datamodule.STDS_NPY_FPATH)).to(self.device)
            # Обратная нормализация
            logits = (logits * self.stds) + self.means
            x_or = (x_or * self.stds) + self.means
        # Вычисляем PCK
        self.train_pck_20(preds=logits, target=x_or)
        self.train_pck_auc_20_40(preds=logits, target=x_or)
        # Логируем метрики
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_PCK_20', self.train_pck_20, on_step=True, on_epoch=False)
        self.log('train_PCK_AUC_20-40', self.train_pck_auc_20_40, on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_end(self):
        mean_epoch_loss = torch.stack(self.train_step_losses).mean()
        self.logger.experiment.add_scalars("losses", {"train_loss" : mean_epoch_loss}, global_step=self.current_epoch)
        self.train_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        # Извлекаем дату из батчей
        _, x_or, x_masked, scores, masked_frames_idx = batch
        # Forward pass
        (logits, theta, beta, _, _, _, _, _, _) = self(x_masked)
        # Лосс применяется только на замаскированные данные, мы же BERT юзаем
        valid_idx = torch.where(masked_frames_idx != -1.)
        logits = logits[valid_idx]
        x_or = x_or[valid_idx]
        scores = scores[valid_idx]
        # Вычисляем потери реконструкции(предсказания)(LRec) и рег лосс(LReg)
        # Вычисляем L1 норму тут ес чо
        lrec = torch.norm(logits[scores > self.eps] - x_or[scores > self.eps], p=1, dim=1).sum()
        beta_t_minus_one = torch.roll(beta, shifts=1, dims=1)
        beta_t_minus_one[:, 0] = 0
        # Вычисляем изменение между текущим beta и предыдущим, умножая на коэффициент регуляризации
        lreg = torch.norm(theta, p=2) + self.weight_beta * torch.norm(beta, p=2) + \
            self.weight_delta * torch.norm(beta - beta_t_minus_one, p=2)
        # Комбинируем оба лосса
        loss = lrec + (self.lmbd * lreg)
        # Аппендим лосс шага, детач нужен чтобы "открепить" данные от общего пула
        # И сделать так, чтобы они не влияли на backprop
        self.train_step_losses.append(loss.detach().cpu())
        # Проверка на нормализованные данные
        if self.normalize_inputs:
            if not hasattr(self, 'means') or not hasattr(self, 'stds'):
                self.means = torch.from_numpy(np.load(self.trainer.datamodule.MEANS_NPY_FPATH)).to(self.device)
                self.stds = torch.from_numpy(np.load(self.trainer.datamodule.STDS_NPY_FPATH)).to(self.device)
            # Обратная нормализация
            logits = (logits * self.stds) + self.means
            x_or = (x_or * self.stds) + self.means
        # Вычисляем PCK
        self.train_pck_20(preds=logits, target=x_or)
        self.train_pck_auc_20_40(preds=logits, target=x_or)
        # Логируем метрики
        self.log('val_loss', loss, on_step=False, prog_bar=True)
        self.log('val_PCK_20', self.train_pck_20, on_step=False, on_epoch=False, prog_bar=True)
        self.log('val_PCK_AUC_20-40', self.train_pck_auc_20_40, on_step=False, on_epoch=False)
        self.log("hp_metric", self.val_pck_20, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        mean_epoch_loss = torch.stack(self.val_step_losses).mean()
        self.logger.experiment.add_scalars("losses", {"val_loss" : mean_epoch_loss}, global_step=self.current_epoch)
        self.val_step_losses.clear()

    def configure_optimizers(self):
        toret={}
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        if self.use_onecycle_lr:
            lr_scheduler_config = dict(
                sheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr = self.lr,
                    total_steps=self.total_steps,
                    pct_start=self.pct_start,
                    anneal_strategy='linear',
                )
            )
        toret['optimizer'] = optimizer
        if self.use_onecycle_lr:
            toret['lr_scheduler'] = lr_scheduler_config

        return toret
