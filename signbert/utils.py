import json
import gc
import torch


def my_import(name):
    """
    Функция для динамического импорта модуля или объекта

    Получает на вход полное имя модуля, который надо импортировать в качестве строки
    эта функция импортирует модуль и возвращает этот модуль или объект

    Params:
    name(str): Полное имя модуля для импорта, например 'numpy.array'

    Returns:
    module/object: Модуль или объект, который был испортирован
    """
    # Сплитуем название модуля и объекта
    components = name.split('.')
    mod = __import__(components[0])
    # Проходимся по модулю иерархически, чтобы получить желаемый нами модуль
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod


def read_json(fpath):
    with open(fpath, 'r') as fid:
        data = json.load(fid)
    return data


def read_txt_as_list(fpath):
    with open(fpath, 'r') as fid:
        data = fid.read().split('\n')
    return data


def dict_to_json_file(dict, fpath):
    with open(fpath, 'w') as fid:
        json.dump(dict, fid)


def num_active_conda_tensors():
    """
    Returns:
    Возвращает количество тензоров которые находится в девайсе
    """
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == "cuda":
                count += 1
        except:
            pass

    return count

