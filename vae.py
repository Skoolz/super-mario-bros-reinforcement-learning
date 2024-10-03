import torch
from torch import nn
from pstd.sd.encoder import VAE_Encoder
from pstd.sd.decoder import VAE_Decoder
from torch.nn import functional as F




class VAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

    def vae_loss(self, recon_x, x, mu, logvar):
        # Ошибка реконструкции: MSE или BCE
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')  # можно использовать BCE, в зависимости от задачи
        # KL-дивергенция
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Нормализация по размеру батча
        batch_size = x.size(0)
        return (recon_loss + kl_loss) / batch_size

    
    def forward(self,x:torch.tensor, training=False, noise:torch.tensor=None):

        input_x = x

        batch_size,channels,height,width = x.shape

        if(noise is None):
            noise = torch.rand(size=(batch_size,4,height//8,width//8),device=x.device,dtype=x.dtype)

        x,mean,log_var = self.encoder.forward(x,noise=noise)

        x = self.decoder.forward(x)

        if(training):
            loss = self.vae_loss(x,input_x,mean,log_var)

            return x,loss

        return x
            