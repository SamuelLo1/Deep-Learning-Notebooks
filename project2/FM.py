import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalFM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels,
            self.modelconfig.num_feat,
            self.modelconfig.num_classes,
            self.modelconfig.input_dim,
        )

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
        B, C, H, W = images.shape
        
        # Step 1: Sample random noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(images)
        x_1 = images
        
        # Step 2: Sample continuous time t ~ Uniform(0, 1)
        t = torch.rand(B, 1, device=images.device)  # Shape: (B, 1)
        
        # Step 3: Linear interpolation - the straight path from noise to data
        # x_t = (1-t)x_0 + t*x_1
        x_t = (1 - t).view(B, 1, 1, 1) * x_0 + t.view(B, 1, 1, 1) * x_1
        
        # Step 4: Compute velocity - the direction we want to move
        # u_t = x_1 - x_0 (constant for linear paths)
        u_t = x_1 - x_0
        
        # Step 5: Classifier-free guidance - randomly drop conditions
        # With probability p_uncond, set condition to null (zeros)
        p_uncond = getattr(self.modelconfig, 'p_uncond', 0.1)
        mask = torch.rand(B, device=images.device) < p_uncond
        conditions_guided = conditions.clone()
        conditions_guided[mask] = self.modelconfig.num_classes  # Use class index for null condition
        
        # Convert to one-hot encoding
        conditions_onehot = F.one_hot(
            conditions_guided, 
            num_classes=self.modelconfig.num_classes + 1
        ).float()[:, :-1]  # Remove the extra dimension from null class
        
        # Step 6: Network predicts velocity at (x_t, c, t)
        velocity_pred = self.network(x_t, t, conditions_onehot)
        
        # Step 7: Compute loss - MSE between predicted velocity and actual velocity
        loss = self.loss_fn(velocity_pred, u_t)

        # ==================================================== #
        return loss

    def sample(self, conditions, omega):
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
        #       generated_images: samples from the model, shape (B,1,28,28)

        B = conditions.shape[0]
        device = next(self.network.parameters()).device
        
        # Step 1: Start with pure noise x_0 ~ N(0, I)
        x = torch.randn(B, 1, 28, 28, device=device)
        
        # Step 2: Integrate from t=0 to t=1 using learned velocity field
        num_steps = 20  # Number of integration steps (higher = better quality, slower)
        dt = 1.0 / num_steps
        with torch.no_grad():
            for i in range(num_steps):
                # Current time step
                t = torch.full((B, 1), i * dt, device=device)
                
                # Get conditional velocity prediction (with class info)
                conditions_onehot = F.one_hot(
                    conditions, 
                    num_classes=self.modelconfig.num_classes
                ).float()
                v_cond = self.network(x, t, conditions_onehot)
                
                # Get unconditional velocity prediction (no class info)
                uncond_onehot = torch.zeros(
                    (B, self.modelconfig.num_classes), 
                    device=device
                )
                v_uncond = self.network(x, t, uncond_onehot)
                
                # Classifier-free guidance blending
                # v = v_uncond + omega * (v_cond - v_uncond)
                # This blends toward conditional when omega > 1
                v = v_uncond + omega * (v_cond - v_uncond)
                
                # Euler integration step: x_{t+dt} = x_t + v(x_t, c, t) * dt
                x = x + v * dt
            
        # ==================================================== #
        generated_images = (x * 0.3081 + 0.1307).clamp(0, 1)
        return generated_images