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

from vae import VAE

# Константы
DATASET_FOLDER = 'game_images_dataset'  # Папка для хранения данных
DATASET_SIZE = 1000  # Количество изображений в датасете
SEQUENCE_LENGTH = 20  # Длина последовательности
IMAGE_HEIGHT = 32  # Установите нужные размеры для изображений
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 4
DATASET_FILENAME = 'game_dataset_gen.npy'  # Имя файла для хранения всех изображений и действий
BATCH_SIZE = 32  # Размер батча для DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONTEXT_SPACE = 8

# Функция для создания папки, если ее нет
def ensure_dataset_folder_exists():
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

# Кастомный PyTorch датасет
class CustomDataset(Dataset):
    def __init__(self, dataset_file, transform=None):
        self.data = np.load(dataset_file, allow_pickle=True)  # Теперь загружаем последовательности кадров и действий
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sequence_data = self.data[idx]
        images = sequence_data['images']
        actions = sequence_data['actions']
        
        if self.transform:
            images = self.transform(images)
        
        return images, torch.tensor(actions, dtype=torch.long)  # Возвращаем кадры и действия

# collate_fn для обработки батчей
def collate_fn(batch):
    images_batch, actions_batch = zip(*batch)
    
    images = np.stack(images_batch, axis=0)  # Из списка последовательностей изображений создаем 5D numpy массив (batch_size, sequence_length, height, width, channels)
    images = torch.tensor(images, dtype=torch.float32) #(B,Seq,C,H,W)
    
    # Преобразуем действия в тензор
    actions = torch.stack(actions_batch) #(B,Seq)
    
    return images, actions

def preprocess_image(image):
    if type(image) is not torch.Tensor:
        image = torch.tensor(image.copy(),dtype=torch.float32)
    image = image.unsqueeze(0)
    image = image.permute(0,3,1,2) #(B,C,H,W)
    image = F.interpolate(image, size=(256, 256), mode='nearest')  # Изменяем размер изображений
    image = rescale(image,(0,255),(-1,1),False)

    return image

def encode_image(image,vae):
    with torch.no_grad():
        vae.eval()
        image = preprocess_image(image) #(3,256,256)
        image = vae.encode(image) #(1,4,32,32)
    return image[0]


class Adv_SMB(SMB):
    def __init__(self, env, model,vae):
        super().__init__(env, model)
        self.vae = vae
    
    def collect_game_dataset(self, total_sequences=DATASET_SIZE, sequence_length=SEQUENCE_LENGTH, deterministic=False):
        ensure_dataset_folder_exists()
        sequence_counter = 0  # Счетчик последовательностей

        # Инициализируем пустой массив для хранения последовательностей
        all_data = []  # Для хранения последовательностей кадров и действий

        action_pad_index = self.env.action_space.n

        # Цикл продолжается, пока не будет собрано достаточно данных
        with tqdm(total=total_sequences, desc="Collecting dataset") as pbar:
            while sequence_counter < total_sequences:
                states = self.env.reset()  # Получение информации при reset
                done = False

                info = env_wrap.reset_infos[0]
                action_buffer = []
                image_buffer = []

                # Инициализируем буфер кадров и действий
                for _ in range(sequence_length):
                    if 'game_screen' in info:
                        game_screen = info['game_screen']
                        game_screen = encode_image(game_screen,self.vae).to('cpu') # in case if we are using vae on cuda
                        if game_screen.shape == (IMAGE_CHANNELS,IMAGE_HEIGHT, IMAGE_WIDTH):
                            image_buffer.append(game_screen)
                        else:
                            image_buffer.append(image_buffer[-1])  # Копируем последний кадр
                    else:
                        image_buffer.append(image_buffer[-1])  # Копируем последний кадр
                    
                    action_buffer.append(action_pad_index)  # Добавляем действие заглушку

                while not done:
                    action, _ = self.model.predict(states, deterministic=deterministic)
                    states, _, done, info = self.env.step(action)

                    info = info[0]
                    
                    # Получаем изображение игры из info
                    if 'game_screen' in info:
                        game_screen = info['game_screen']
                        game_screen = encode_image(game_screen,self.vae).to('cpu') # in case if we are using vae on cuda
                        if game_screen.shape == (IMAGE_CHANNELS,IMAGE_HEIGHT, IMAGE_WIDTH):
                            image_buffer.append(game_screen)
                            action_buffer.append(action[0])
                            
                            # Удаляем самый старый кадр и действие из буфера
                            image_buffer.pop(0)
                            action_buffer.pop(0)
                            
                            if len(image_buffer) == sequence_length:
                                all_data.append({'images': torch.stack(image_buffer,dim=0).numpy(), 'actions': np.array(action_buffer)})
                                sequence_counter += 1
                                pbar.update(1)
                            
                                if sequence_counter >= total_sequences:
                                    break

        # Сохраняем данные после завершения сбора
        np.save(os.path.join(DATASET_FOLDER, DATASET_FILENAME), all_data)
        print(f"Dataset collection finished. Total sequences collected: {sequence_counter}/{total_sequences}")


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
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env_wrap,device='cpu')

    vae = VAE()

    vae.load_state_dict(torch.load('vae_model.pth',map_location=DEVICE))

    s = Adv_SMB(env=env_wrap, model=model,vae=vae)
    s.collect_game_dataset(total_sequences=100)
