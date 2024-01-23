import torch
import numpy as np


class DDPM:

    def __init__(self, generator: torch.Generator,                                                                          #pytorch random number generator
                  num_training_steps=1000,                                                                                  # training steps (noising and denoising steps)
                  beta_start: float = 0.00085,                                                                              # beta start
                    beta_end: float = 0.0120):                                                                              # beta end schedule
        

                                                                                                                             # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
                                                                                                                             # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
       
        
        
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2         # beta schedule 
        
        self.alphas = 1.0 - self.betas                                                                                        # alpha schedule is 1 betas
        
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)                                                               # cumprod is cumulative product it works by multipling the current postion in the vector by all the previous entries in the vector ie alph= [ .99 , .98 ,... ]  alpha cumprod= [ .99 , .99*.98  ,  .99*.98* .....]
        
        self.one = torch.tensor(1.0)


        self.generator = generator                                                                                             # random number generator can be set with a seed if need most of the time not needed

        self.num_train_timesteps = num_training_steps                                                                           # number of steps  to train 
       
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())                                       
  
    def set_inference_timesteps(self, num_inference_steps=50):                                                                  #during inference timestpes are different for speed
        self.num_inference_steps = num_inference_steps                                                                          # there fore we can change the steps size
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)                      # re calculating the timesteps 
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:                                                                     
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

                                                                                                                                # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
                                                                                                                                # and sample from it to get previous sample
                                                                                                                                # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

                                                                                                                                  # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength=1):
        
                                                                                                                            # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)                                    # during inference if there is an image input we can encode the image while addin noise and pass it through the unet to get the undet to edit the image (mostly inpainting)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise
        
         
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample                                                                                               # since the unet only predicts the noise we need to remove the image there for we use the coefficients generate the previous sample in the tie steps for the unet to run over again
    
    def add_noise(                                                                                                            # simple forward diffusion according to the paper, it introduces noise to a sample image according to its position in the time steps if the image is the inital image almost no noise if further alot of noise
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

        
    def time_embedding(timestep):
    
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    