import torch
import torch.nn as nn
from tqdm.auto import tqdm
import math

class Discretised():
    # K: number of Bins
    # k: {1, 2, ..., K} 1-index
    def __init__(self, K: int):
        self.K = K
    def center(self, k: int):
        return (2 * k - 1) / self.K - 1
    def left(self, k: int):
        return self.center(k) - 1 / self.K
    def right(self, k: int):
        return self.center(k) + 1 / self.K
    def find_nearest_index(self, x):
        # x: (B, C, H, W)
        values = (2 * torch.arange(1, self.K+1, device=x.device) - 1) / self.K - 1 # values: (K)
        differences = torch.abs(values.view(1, 1, 1, 1, self.K) - x.unsqueeze(-1))
        return torch.argmin(differences, dim=-1) + 1
    def get_center(self, x):
        nearest_index = self.find_nearest_index(x)
        return self.center(nearest_index)

class BFNDiscretisedData(nn.Module):
    def __init__(self, unet, K: int, in_channels=3, sigma=math.sqrt(0.001), device='cuda'):
        super(BFNDiscretisedData, self).__init__()
        self.sigma = torch.tensor(sigma, device=device)
        self.in_channels = in_channels
        self.unet = unet.to(device)
        self.K = K
        self.discretised = Discretised(K)
        self.Kl = self.discretised.left(torch.arange(1, self.K+1).to(device)) # (K, )
        self.Kr = self.discretised.right(torch.arange(1, self.K+1).to(device))
        self.Kc = self.discretised.center(torch.arange(1, self.K+1).to(device))
    def forward(self, theta, t, ema=None):
        """
        Forward pass of the Bayesian Flow Network.

        Parameters
        ----------
        theta : torch.Tensor
            Tensor of shape (B, C, H, W).
        t : torch.Tensor
            Tensor of shape (B,).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, 2*C, H, W).
        """
        t = self.get_time_embedding(t)
        if ema is None:
            output = self.unet(theta, t)
        else:
            with ema.average_parameters():
                output = self.unet(theta, t)
        return output

    def get_time_embedding(self, timestep):
        # timestep: (bs, )
        # output: (bs, 320)
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32, device=timestep.device) / 160)
        x = timestep[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def discretised_cdf(self, x, mu, sigma):
        """
        Parameters ( arbitrary shape X )
        ----------
        x: torch.Tensor
          Tensor of shape X
        mu: torch.Tensor
          Tensor of shape X
        sigma: torch.Tensor
          Tensor of shape X
        Return
        ------
        torch.Tensor
            Output tensor of shape X
        """
        g = 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))
        g = torch.where(x >= 1, torch.ones_like(x), g)
        g = torch.where(x <= -1, torch.zeros_like(x), g)
        return g
    
    def discretised_output_distribution(self, mu, t, gamma, ema=None, t_min=1e-10):
        """
        Parameters
        ----------
        mu: torch.Tensor
          Tensor of shape (B, C, H, W)
        t: torch.Tensor
          Tensor of shape (B, )
        gamma: torch.Tensor
          Tensor of shape (B, )
        Return
        ------
        torch.Tensor
            Output tensor of shape (B, C, H, W, K)
        """
        output = self.forward(mu, t, ema=ema)
        _, c, _, _ = output.shape
        mu_eps, ln_sigma_eps = output[:, :c//2, :, :], output[:, c//2:, :, :]
        mask = t < t_min                                                              # want t=0 ⇨ mux=0 sigmax=1
        gamma = torch.where(mask[:, None, None, None], torch.ones_like(gamma), gamma) # t=0 ⇨ gammma=1 ⇨ gam=0 mu=0 mux=0 sigmax=1 
        gam = ((1 - gamma) / gamma).sqrt()
        mux = mu / gamma - gam * mu_eps
        sigmax = torch.where(mask[:, None, None, None], torch.ones_like(ln_sigma_eps), gam * torch.exp(ln_sigma_eps))
        output = self.discretised_cdf(self.Kr.reshape(1, 1, 1, 1, -1), mux.unsqueeze(-1), sigmax.unsqueeze(-1)) - self.discretised_cdf(self.Kl.reshape(1, 1, 1, 1, -1), mux.unsqueeze(-1), sigmax.unsqueeze(-1))
        return output
    
    def process_infinity(self, x, t=None, return_params=False):
        """
        x : torch.Tensor
            Tensor of shape (B, C, H, W).
        t : torch.Tensor
            Tensor of shape (B, ).
        Returns
        -------
        torch.Tensor
            Output tensor of shape (1, ).
        """
        x = self.discretised.get_center(x)
        bs, c, h, w = x.shape
        if t is None:
            t = torch.rand((bs, ), device=x.device)
        gamma = 1 - self.sigma ** (2 * t[:, None, None, None])
        mu = gamma * x + (gamma * (1 - gamma)).sqrt() * torch.randn_like(x)
        p_output = self.discretised_output_distribution(mu, t, gamma) #(B, C, H, W, K)
        k_hat = torch.matmul(p_output, self.Kc) #(B, C, H, W)
        L_infinity = -self.sigma ** (-2 * t[:, None, None, None]) * torch.log(self.sigma) * (x - k_hat)**2
        if return_params:
            return L_infinity.mean(), mu, self.discretised.get_center(k_hat), t
        else:
            return L_infinity.mean()
        
    def process_discrete(self, x, step=None, max_step=1000, return_params=False):
        """
        x : torch.Tensor
            Tensor of shape (B, C, H, W).
        t : torch.Tensor
            Tensor of shape (B, ).
        Returns
        -------
        torch.Tensor
            Output tensor of shape (1, ).
        """
        x = self.discretised.get_center(x)
        bs, c, h, w = x.shape
        if step is None:
            step = torch.randint(1, max_step+1, (bs, ), device=x.device)
        t = (step - 1) / max_step
        gamma = 1 - self.sigma ** (2 * t[:, None, None, None])
        mu = gamma * x + (gamma * (1 - gamma)).sqrt() * torch.randn_like(x)
        p_output = self.discretised_output_distribution(mu, t, gamma) #(B, C, H, W, K)
        k_hat = torch.matmul(p_output, self.Kc) #(B, C, H, W)
        alpha = self.sigma ** (-2 * step[:, None, None, None] / max_step) * (1 - self.sigma ** (2 / max_step))
        y = x + torch.randn_like(x) / alpha.sqrt()
        L_discrete  = -alpha / 2 * (y - x) ** 2 
        L_discrete -= (p_output * (-alpha.unsqueeze(-1) / 2 * (y.unsqueeze(-1) - self.Kc.reshape(1, 1, 1, 1, -1)) ** 2).exp()).sum(-1).log()
        L_discrete *= max_step
        if return_params:
            return L_discrete.mean(), mu, self.discretised.get_center(k_hat), t
        else:
            return L_discrete.mean()
        

    @torch.inference_mode()
    def sample(self, h, w, batch_size=4, steps=1000, ema=None, return_samples=None, device="cpu"):
        self.eval()
        mu = torch.zeros((batch_size, self.in_channels, h, w), device=device)
        rho = 1
        ret_list = []
        for step in tqdm(range(1, steps+1)):
            t = (step - 1) / steps * torch.ones((batch_size, ), device=device)
            k = self.discretised_output_distribution(mu, t, 1 - self.sigma ** (2 * t[:, None, None, None]), ema=ema)
            k = torch.matmul(k, self.Kc)
            k = self.discretised.get_center(k)
            alpha = self.sigma ** (-2 * step / steps) * (1 - self.sigma ** (2 / steps))
            y = k + torch.randn_like(k) / alpha.sqrt()
            mu = (rho * mu + alpha * y) / (rho + alpha)
            rho += alpha
            if return_samples is not None and (step-1) % return_samples == 0:
                ret_list.append(k)
        t = torch.ones((batch_size, ), device=device)
        k = self.discretised_output_distribution(mu, t, 1 - self.sigma ** (2 * t[:, None, None, None]), ema=ema)
        k = torch.matmul(k, self.Kc)
        k = self.discretised.get_center(k)
        if return_samples is None:
            return k
        ret_list.append(k)
        return k, ret_list