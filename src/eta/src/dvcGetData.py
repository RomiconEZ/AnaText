import configparser
import os
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd


def get_dvc_remote_details(config_path=".dvc/config"):
    """
    Получает информацию о DVC remote из конфигурационного файла.

    Аргументы:
    config_path (str): путь к конфигурационному файлу DVC. По умолчанию '.dvc/config'.

    Возвращает:
    tuple: Кортеж, содержащий имя remote и его URL, или (None, None) в случае ошибки.

    Описание:
    Функция читает конфигурационный файл DVC, указанный в config_path, для извлечения информации
    о текущем remote хранилище. Она возвращает имя remote хранилища и его URL.
    В случае отсутствия секций или опций в конфигурации, функция возвращает (None, None) и выводит
    соответствующее сообщение об ошибке.
    """
    current_path = Path(__file__).parent
    filename = str(current_path.parent.parent.parent / config_path)

    config = configparser.ConfigParser()
    config.read(filename)

    try:
        # Получение имени remote из секции [core]
        remote_name = config.get("core", "remote")
        # Получение URL для этого remote
        remote_url = config.get(f"'remote \"{remote_name}\"'", "url")

        return str(remote_name), str(remote_url)
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Ошибка при извлечении данных: {e}")
        return None, None


def load_data(datapath):
    """
    Загружает данные из файла, используя DVC API.

    Аргументы:
    datapath (str): путь к файлу данных относительно DVC хранилища.

    Возвращает:
    DataFrame или ndarray: загруженные данные в соответствующем формате,
                           или None в случае ошибки.

    Описание:
    Функция определяет формат файла данных на основе его расширения и пытается загрузить данные
    с использованием DVC API. Для этого она использует информацию о remote DVC хранилище, полученную
    из функции get_dvc_remote_details. Поддерживаются форматы CSV, JSON, Excel, Parquet, Numpy.
    В случае ошибки при загрузке данных или неподдерживаемого формата файла функция возвращает None
    и выводит соответствующее сообщение об ошибке.
    """
    # Определение формата файла по расширению
    file_extension = os.path.splitext(datapath)[1].lower()

    # Получение данных о remote
    remote_name, remote_url = get_dvc_remote_details()
    if not (remote_name and remote_url):
        print("Ошибка в настройках DVC.")
        return None
    print(f"Имя remote: {remote_name}, URL: {remote_url}")
    try:
        with dvc.api.open(
            path=datapath,
            repo=remote_url,
            remote=remote_name,
            mode="rb",  # Открыть в двоичном режиме для универсальности
        ) as fd:
            if file_extension == ".csv":
                return pd.read_csv(fd)
            elif file_extension == ".json":
                return pd.read_json(fd)
            elif file_extension == ".xlsx":
                return pd.read_excel(fd)
            elif file_extension == ".parquet":
                return pd.read_parquet(fd)
            elif file_extension in [".npy", ".npz"]:
                return np.load(fd, allow_pickle=True)
            else:
                print(f"Формат файла {file_extension} не поддерживается.")
                return None
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None
