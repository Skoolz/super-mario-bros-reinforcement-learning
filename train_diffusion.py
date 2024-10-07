import os
import torch
import torch.optim as optim
from tqdm import tqdm
from pstd.sd.ddpm import DDPMSampler
from pstd.sd.diffusion import Diffusion
from collect_data_gen import create_dataloader
from collect_data_gen import SEQUENCE_LENGTH,CONTEXT_SPACE

# Гиперпараметры
DATASET_FOLDER = "game_images_dataset"
DATASET_FILENAME = "game_dataset_gen.npy"
BATCH_SIZE = 25  # Малый батч для градиентного накопления
LEARNING_RATE = 1e-6
EPOCHS = 100
NUM_TRAINING_STEPS = 1000
ACCUMULATION_STEPS = 2  # Количество шагов для накопления градиентов
INTERMEDIATE_SAVE_FREQUENCY = 5  # Частота сохранения модели
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'diffusion_model.pth'
CHECKPOINT_PATH = 'diffusion_checkpoint.pth'


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


# Функция обучения с накоплением градиентов и промежуточными сохранениями
def train_diffusion_model(model, dataloader, sampler,checkpoint_path=None):
    

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Загрузка чекпоинта, если есть
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        optimizer.zero_grad()  # Инициализация градиентов
        for step, batch in enumerate(pbar):
            images, actions = batch

            #images (batch_size,seq_len,4,32,32)
            #actions (batch_size,seq_len)

            batch_size, seq_len, channels, height, width = images.shape

            # Извлекаем последние кадры для предсказания (target)
            target_images = images[:, -1]  # Target кадры #(b,4,32,32)
            
            # Убираем последний кадр из исходных данных
            images = images[:, :-1]  # Все кадры кроме последнего (b,seq_len-1,4,32,32)
            
            # Объединяем оставшиеся кадры (batch_size, (seq_len-1)*4, 32, 32)
            images = images.view(batch_size, -1, height, width)  # (batch_size, (seq_len-1)*4, 32, 32)

            # Получаем случайные временные шаги для каждого батча
            timesteps = torch.randint(0, sampler.num_train_timesteps, (batch_size,), device=DEVICE)

            # Добавляем шум только к target кадрам
            noisy_target, noise = sampler.add_noise(target_images, timesteps)
            
            # Объединяем оставшиеся кадры и зашумленные target кадры
            combined_images = torch.cat((images, noisy_target), dim=1)
            
            # Получаем time embedding для каждого элемента в батче
            time_embedding = sampler.get_time_embedding(timesteps)
            
            # Перемещаем данные на GPU (если доступно)
            combined_images = combined_images.to(DEVICE) #(batch_size, seq_len*4, 32, 32)
            actions = actions.to(DEVICE) #(batch_size, seq_len)
            noise = noise.to(DEVICE) #(batch_size, 4, 32, 32)
            time_embedding = time_embedding.to(DEVICE) #(batch_size, 320)

            # Прогоняем через модель
            predicted_noise = model(combined_images, actions, time_embedding) #(batch_size, 4, 32, 32)

            # Вычисляем ошибку (MSE между реальным и предсказанным шумом для target)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            loss = loss / ACCUMULATION_STEPS  # Делим на количество шагов накопления
            epoch_loss += loss.item()

            # Обратное распространение ошибки
            loss.backward()

            # Выполняем шаг оптимизатора только после нескольких шагов (ACCUMULATION_STEPS)
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            # Обновление отображения tqdm
            pbar.set_postfix({"Loss": epoch_loss / (pbar.n + 1)})
        
        print(f"Epoch {epoch+1} finished with loss: {epoch_loss / len(dataloader)}")

        # Сохраняем модель каждые INTERMEDIATE_SAVE_FREQUENCY эпох
        if (epoch + 1) % INTERMEDIATE_SAVE_FREQUENCY == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # Сохранение финальной модели
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved to", MODEL_SAVE_PATH)

# Пример использования
if __name__ == "__main__":
    # Инициализация модели, оптимизатора и семплера
    model = Diffusion(seq_len=SEQUENCE_LENGTH,context_space=CONTEXT_SPACE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Инициализация DDPMSampler
    generator = torch.Generator()
    sampler = DDPMSampler(generator=generator, num_training_steps=NUM_TRAINING_STEPS)

    # Загрузка данных
    dataloader = create_dataloader(os.path.join(DATASET_FOLDER, DATASET_FILENAME), batch_size=BATCH_SIZE, shuffle=True)
    
    
    
    # Запуск обучения
    train_diffusion_model(model, dataloader, sampler,checkpoint_path=CHECKPOINT_PATH)
