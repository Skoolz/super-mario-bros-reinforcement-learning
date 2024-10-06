import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from gym_utils import SMB

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import os

from gym_utils import SMBRamWrapper, load_smb_env, SMB

from pstd.sd.pipeline import rescale

from torch.nn import functional as F

# Константы
DATASET_FOLDER = 'game_images_dataset'  # Папка для хранения данных
DATASET_SIZE = 1000  # Количество изображений в датасете
IMAGE_HEIGHT = 240  # Установите нужные размеры для изображений
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
DATASET_FILENAME = 'game_dataset.npy'  # Имя файла для хранения всех изображений
BATCH_SIZE = 32  # Размер батча для DataLoader

# Функция для создания папки, если ее нет
def ensure_dataset_folder_exists():
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

# Кастомный PyTorch датасет
class CustomDataset(Dataset):
    def __init__(self, dataset_file, transform=None):
        self.data = np.load(dataset_file)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image  # Здесь возвращаем пока как numpy массив для дальнейшей обработки в collate_fn

# collate_fn для обработки батчей
def collate_fn(batch):
    images = np.stack(batch, axis=0)  # Из списка изображений создаем 4D numpy массив (batch_size, height, width, channels)
    images = torch.tensor(images, dtype=torch.float32)  # Преобразуем в PyTorch тензор
    
    images = images.permute(0,3,1,2) #(B,C,H,W)

    images = F.interpolate(images, size=(256, 256), mode='nearest')
    
    images = rescale(images,(0,255),(-1,1),False)
    return images


class Adv_SMB(SMB):
    def __init__(self, env, model):
        super().__init__(env, model)
    # Функция для сбора датасета
    def collect_game_dataset(self, total_images=DATASET_SIZE, deterministic=False):
        ensure_dataset_folder_exists()
        image_counter = 0  # Счетчик изображений

        # Инициализируем пустой массив для хранения всех изображений
        all_images = np.zeros((total_images, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)

        # Цикл продолжается, пока не будет собрано достаточно данных
        with tqdm(total=total_images, desc="Collecting dataset") as pbar:
            while image_counter < total_images:
                states = self.env.reset()  # Получение информации при reset
                done = False

                info = env_wrap.reset_infos[0]
                
                if 'game_screen' in info:
                        game_screen = info['game_screen']

                        # Проверка формы изображения
                        if game_screen.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
                            all_images[image_counter] = game_screen
                            image_counter += 1
                            pbar.update(1)  # Обновляем прогресс

                # Прерываем внутренний цикл, если собрано достаточно изображений
                if image_counter >= total_images:
                    break

                while not done:
                    action, _ = self.model.predict(states, deterministic=deterministic)
                    states, reward, done, info = self.env.step(action)

                    info = info[0]

                    # Получаем изображение игры из info
                    if 'game_screen' in info:
                        game_screen = info['game_screen']

                        # Проверка формы изображения
                        if game_screen.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
                            all_images[image_counter] = game_screen
                            image_counter += 1
                            pbar.update(1)  # Обновляем прогресс

                    # Прерываем внутренний цикл, если собрано достаточно изображений
                    if image_counter >= total_images:
                        break

        # Сохраняем массив после завершения сбора данных
        np.save(os.path.join(DATASET_FOLDER, DATASET_FILENAME), all_images)
        print(f"Dataset collection finished. Total images collected: {image_counter}/{total_images}")


# Создание DataLoader
def create_dataloader(dataset_file, batch_size=BATCH_SIZE, shuffle=True):
    dataset = CustomDataset(dataset_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    MODEL_DIR = './models'

    # obs = 4 frames
    crop_dim = [0, 16, 0, 13]
    n_stack = 4
    n_skip = 4
    MODEL_NAME = 'pre-trained-1'

    env_wrap = load_smb_env('SuperMarioBros-1-1-v0', crop_dim, n_stack, n_skip)
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env_wrap)


    s = Adv_SMB(env=env_wrap,model=model)
    s.collect_game_dataset() 