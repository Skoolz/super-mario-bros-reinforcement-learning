from diffusers import AutoencoderKL
import torch
from PIL import Image
import numpy as np

vae = AutoencoderKL.from_pretrained('./vae',use_safetensors=False)

image = Image.open("World_1-1_Super_Mario_Bros.png").convert("RGB")
image = image.resize((512, 512))  # Изображения должны быть размера 512x512 для модели
image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
image_tensor = image_tensor * 2 - 1  # Нормализация в диапазон [-1, 1]

# Кодируем изображение в латентное пространство
with torch.no_grad():
    latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215

print(latent.shape)

# Декодируем латентное представление обратно в изображение
with torch.no_grad():
    decoded_image = vae.decode(latent / 0.18215)
    decoded_image = decoded_image.sample

# Преобразуем результат в изображение
decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
decoded_image = decoded_image.squeeze().permute(1, 2, 0).numpy()
decoded_image = (decoded_image * 255).astype(np.uint8)
decoded_image = Image.fromarray(decoded_image)
decoded_image = decoded_image.resize((256,256))

# Сохраняем или отображаем результат
decoded_image.save("reconstructed_image.jpg")
decoded_image.show()