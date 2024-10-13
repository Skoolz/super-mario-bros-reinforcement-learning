import pygame
import numpy as np
from collect_data_gen import CustomDataset
import os
import torch
from collections import deque
from pipeline_gen import generate,rescale
from vae import VAE
from pstd.sd.diffusion import Diffusion

# Константы
SCREEN_WIDTH = 800  # Ширина окна
SCREEN_HEIGHT = 600  # Высота окна
FRAME_WIDTH = 256  # Ширина кадра
FRAME_HEIGHT = 256  # Высота кадра
FPS = 60  # Количество кадров в секунду

DATASET_FOLDER = 'game_images_dataset'  # Папка для хранения данных
DATASET_FILENAME = 'game_dataset_gen.npy'  # Имя файла для хранения всех изображений и действий

class Game:
    def __init__(self, real_time=True,seq_len=20, diffusion=None,vae=None):
        # Инициализация pygame и настройка окна
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Numpy Frame Display")
        self.clock = pygame.time.Clock()
        self.running = True
        self.real_time = real_time  # Параметр, отвечающий за работу в реальном времени
        self.frame_buf_limit = seq_len

        self.frame_buffer = deque(maxlen=self.frame_buf_limit)
        self.action_buffer = deque(maxlen=self.frame_buf_limit)

        self.vae = vae
        self.diffusion = diffusion

        self.data_index = 0

    def generate_frame(self,autoregressive=True):

        if(autoregressive):
            self.frame_buffer.popleft()
        else:
            self.frame_buffer.pop()
        
        images = torch.cat(list(self.frame_buffer),dim=0) #(seq_len-1,4,32,32)
        images = images.unsqueeze(0) #(1,seq_len-1,4,32,32)
        images = images.view(1,-1,32,32) #(1,(seq_len-1)*4,32,32)

        actions = torch.tensor(list(self.action_buffer)) #(seq_len)
        actions = actions.unsqueeze(0) #(1,seq_len)

        image,latent = generate(actions=actions,images=images,diffusion=self.diffusion,
                                vae=self.vae,n_inference_steps=15)
        
        #image (1,256,256,3)
        #latent (1,4,32,32)

        return image,latent
    
    def init_game(self,images,actions,render=True):
        self.frame_buffer = deque(images, maxlen=self.frame_buf_limit)
        self.action_buffer = deque(actions,maxlen=self.frame_buf_limit)

        if(render):
            last_image = self.frame_buffer[-1]
            self.render_frame(last_image,latent=True)

    
    def render_frame(self,image,latent=False,batch_dim=True):
        #image (1,4,32,32) or (256,256,3)

        if(latent):
            image = self.vae.decode(image)
            image = rescale(image, (-1, 1), (0, 255), clamp=True)
            image = image.permute(0, 2, 3, 1)
            image = image.to("cpu", torch.uint8).numpy()

        if(batch_dim):
            image = image[0]
        image = np.transpose(image,(1,0,2))
        
        pygame_frame = pygame.surfarray.make_surface(image)
        scaled_frame = pygame.transform.scale(pygame_frame, (self.screen.get_width(), self.screen.get_height()))
        self.screen.blit(scaled_frame, (0, 0))
        pygame.display.flip()

    def debug_game(self):
        self.data_index+=1

        images,actions = load_state(self.data_index)

        self.init_game(images,actions,render=False)

        image,latent = self.generate_frame(False)

        self.render_frame(image)


    def handle_events(self):
        # Обработка событий (действий игрока)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                # Обновляем размер окна при изменении
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                return True  # Возвращаем True, если было событие изменения окна
            elif event.type == pygame.KEYDOWN:

                if(event.key == pygame.K_RIGHT):
                    #self.action_buffer.append(5)
                
                    #image,latent = self.generate_frame()

                    #self.render_frame(image)

                    #self.frame_buffer.append(latent)
                    pass
                return True  # Возвращаем True, если была нажата клавиша
        return False

    def run(self):
        while self.running:
            # Если игра не в реальном времени, ждем действий игрока
            if not self.real_time:
                event_happened = self.handle_events()
                if not event_happened:
                    continue  # Пропускаем обновление кадра, если нет событий
            else:
                # В реальном времени все равно обрабатываем события, но не ждем их
                self.handle_events()

            # Получение кадра
            #frame = self.generate_frame()

            # Преобразование numpy массива в поверхность pygame
            #pygame_frame = pygame.surfarray.make_surface(frame)

            # Масштабирование изображения до размеров окна
            #scaled_frame = pygame.transform.scale(pygame_frame, (self.screen.get_width(), self.screen.get_height()))

            # Отображение кадра на экране
            #self.screen.blit(scaled_frame, (0, 0))

            # Обновление дисплея
            #pygame.display.flip()

            # Ограничение FPS, если игра в реальном времени
            if self.real_time:
                self.debug_game()
                self.clock.tick(FPS)

        pygame.quit()

def load_state(index=0):
    dataset = CustomDataset(os.path.join(DATASET_FOLDER, DATASET_FILENAME))

    images,actions = dataset[index]

    #images (seq_len,4,32,32)
    #actions (seq_len)

    images = torch.tensor(images,dtype=torch.float32)

    actions = actions.tolist() # [()] * seq_len
    
    seq_len = len(actions)

    images = torch.chunk(images,seq_len,dim=0) #[(1,4,32,32)] * seq_len
    

    return images,actions




if __name__ == "__main__":
    
    
    images,actions = load_state(40)

    vae = VAE()
    vae.load_state_dict(torch.load('vae_model.pth',map_location='cpu'))
    print('VAE was loaded')

    diffusion = Diffusion(seq_len=20,context_space=8)
    diffusion.load_state_dict(torch.load('diffusion_model.pth',map_location='cpu'))
    print('Diffusion was loaded')

    game = Game(real_time=True,seq_len=20,vae=vae,diffusion=diffusion)
    game.init_game(images,actions)
    game.data_index = 40
    game.run()
