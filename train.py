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
BATCH_SIZE = 4
LEARNING_RATE = 0.00001
EPOCHS = 100
INTERMEDIATE_SAVE_FREQUENCY = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'vae_model.pth'
INTERMEDIATE_MODEL_SAVE_PATH = 'vae_model_epoch_{}.pth'




# Определение функции потерь (например, для VAE это может быть рекуррентная ошибка + KL дивергенция)

# Функция обучения
def train_vae(model, dataloader, epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for batch_idx, data in progress_bar:
            # Перемещение данных на устройство
            data = data.to(device)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход через модель
            recon_batch,loss = model.forward(data,training=True)
            
            # Обратное распространение
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
        
        # Сохранение промежуточной модели
        if (epoch + 1) % INTERMEDIATE_SAVE_FREQUENCY == 0:
            torch.save(model.state_dict(), INTERMEDIATE_MODEL_SAVE_PATH.format(epoch + 1))

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss / len(dataloader.dataset):.6f}")

    # Сохранение финальной модели
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved to", MODEL_SAVE_PATH)

# Пример вызова функции

dataloader = create_dataloader(os.path.join(DATASET_FOLDER, DATASET_FILENAME), batch_size=BATCH_SIZE, shuffle=True)

model = VAE().to(DEVICE)

train_vae(model, dataloader, EPOCHS, DEVICE)
