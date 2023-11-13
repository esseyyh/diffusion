import torch


class BetaScheduler:
    def __init__(self,start,end,time_steps, type="linear",) -> None:
        """ creates the betas schedules for adding noise takes in start,end,steps and the type of schedulling  and outputs a vector of schdeules """
        schedule_map = {
            "linear": self.linear_beta_schedule,
            "quadratic": self.quadratic_beta_schedule,
            "cosine": self.cosine_beta_schedule,
            "sigmoid": self.sigmoid_beta_schedule,
        }
        self.schedule = schedule_map[type]
        self.start=start
        self.end=end
        self.step=time_steps

    def __call__(self):
        

        return self.schedule(self.step,self.start,self.end)

    @staticmethod
    def linear_beta_schedule(timesteps,start,end):
        beta_start = start
        beta_end = end
        b=torch.linspace(beta_start, beta_end , timesteps)
        return b

    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
