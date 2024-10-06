import os
import torch
import torch.optim as optim
from tqdm import tqdm
from collect_data import create_dataloader
from vae import VAE
from torch.nn import functional as F

# Константы
DATASET_FOLDER = 'game_images_dataset'  # Папка для хранения данных
DATASET_FILENAME = 'game_dataset.npy'  # Имя файла для хранения всех изображений
BATCH_SIZE = 25
LEARNING_RATE = 0.00001
EPOCHS = 25
INTERMEDIATE_SAVE_FREQUENCY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'vae_model.pth'
CHECKPOINT_PATH = 'vae_checkpoint.pth'
USE_FLOAT16 = False  # Переключатель для использования float16

# Функция для сохранения состояния обучения
def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

# Функция для загрузки состояния обучения
def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path}, starting from epoch {start_epoch+1}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch")
        return 0

# Функция обучения
def train_vae(model, dataloader, epochs, device, use_float16=False, checkpoint_path=None):
    model.to(device)
    
    if use_float16:
        model.half()  # Перевод модели в формат float16
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Если существует чекпоинт, загружаем модель, оптимизатор и номер эпохи
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path) if checkpoint_path else 0

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for batch_idx, data in progress_bar:
            # Перемещение данных на устройство
            data = data.to(device)
            
            if use_float16:
                data = data.half()  # Перевод данных в формат float16
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход через модель
            recon_batch, loss = model.forward(data, training=True)
            
            # Обратное распространение
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
        
        # Сохранение промежуточного состояния модели (чекпоинт)
        if (epoch + 1) % INTERMEDIATE_SAVE_FREQUENCY == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
        
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss / len(dataloader.dataset):.6f}")

    # Сохранение финальной модели
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved to", MODEL_SAVE_PATH)

# Пример вызова функции

dataloader = create_dataloader(os.path.join(DATASET_FOLDER, DATASET_FILENAME), batch_size=BATCH_SIZE, shuffle=True)

train_vae(VAE(), dataloader, EPOCHS, DEVICE, USE_FLOAT16, CHECKPOINT_PATH)
