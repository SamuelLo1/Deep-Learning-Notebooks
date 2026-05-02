import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels, 
            self.modelconfig.num_feat, 
            self.modelconfig.num_classes, 
            self.modelconfig.input_dim
        )

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.modelconfig.beta_1, self.modelconfig.beta_T, self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       

        betas = torch.linspace(beta_1, beta_T, T, device=t_s.device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        idx = t_s.long() - 1

        beta_t = betas[idx].squeeze()
        alpha_t = alphas[idx].squeeze()

        sqrt_beta_t = torch.sqrt(beta_t)
        oneover_sqrt_alpha = 1.0 / torch.sqrt(alpha_t)
        alpha_t_bar = alphas_bar[idx]
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1.0 - alpha_t_bar)

        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function.  
        B = images.shape[0]
        t_s = torch.randint(0, self.modelconfig.T, (B, 1), device=device)

        # set the value of condition to the uncondition value with a prob of dropout
        one_hot_conditions = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float()
        mask_p = 0.1  
        drop_mask = torch.rand(B) < mask_p
        one_hot_conditions[drop_mask] = self.modelconfig.condition_mask_value  
        scheduler_dict = self.scheduler(t_s)

        # sample noise in shape of image
        noise = torch.randn_like(images)

        # proper reshaping for broadcasting
        sqrt_alpha_bar = scheduler_dict['sqrt_alpha_bar'].view(B, 1, 1, 1)
        sqrt_oneminus_alpha_bar = scheduler_dict['sqrt_oneminus_alpha_bar'].view(B, 1, 1, 1)

        # compute noisy images
        noisy_images = sqrt_alpha_bar * images + sqrt_oneminus_alpha_bar * noise
        # predict noise
        predicted_noise = self.network(noisy_images, t_s, one_hot_conditions)
        # compute loss
        noise_loss = self.loss_fn(predicted_noise, noise)


        # ==================================================== #
        return noise_loss

    def sample(self, conditions, omega):
        T = self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  
        B = conditions.shape[0]
        one_hot_conditions = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float()
        one_hot_conditions_uncond = torch.zeros_like(one_hot_conditions)
        
        # starting from pure noise and randomness and iteratively deenoise to get desired conditional
        X_t = torch.randn(B, self.modelconfig.num_channels, self.modelconfig.input_dim, self.modelconfig.input_dim, device=conditions.device)
        
        # progress bar for sampling and iterations
        with torch.no_grad():
            for t in tqdm(range(T, 0, -1), desc='sampling'):
                # get the scheduler dict for current time step
                t_batch = torch.full((B, 1), t, device=conditions.device)
                scheduler_dict = self.scheduler(t_batch)

                # predict noise for both conditional and uncondtional inputs
                predicted_noise_cond = self.network(X_t, t_batch, one_hot_conditions)
                predicted_noise_uncond = self.network(X_t, t_batch, one_hot_conditions_uncond)

                # blend the cond and uncond predictions using omega
                predicted_noise = predicted_noise_uncond + omega * (predicted_noise_cond - predicted_noise_uncond)

                # compute the mean for the reverse process
                alpha_t = scheduler_dict['alpha_t'].view(B, 1, 1, 1)
                sqrt_one_minus_alpha_bar_t = scheduler_dict['sqrt_oneminus_alpha_bar'].view(B, 1, 1, 1)
                sqrt_beta_t = scheduler_dict['sqrt_beta_t'].view(B, 1, 1, 1)
                sqrt_one_over_alpha_t = scheduler_dict['oneover_sqrt_alpha'].view(B, 1, 1, 1)
                
                mean = sqrt_one_over_alpha_t * (X_t - ((1 - alpha_t) / sqrt_one_minus_alpha_bar_t) * predicted_noise)
                
                # 6. Add noise for t > 1, else just take mean
                if t > 1:
                    z = torch.randn_like(X_t)
                    X_t = mean + sqrt_beta_t * z
                else:
                    generated_images = mean

        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1)
        return generated_images