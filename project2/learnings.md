### DDPM

- A DDPM stands for Denoising Diffusion Probabilistic Model.
- The goal of the DDPM uses ResUNet to reverse a noised image to a desired image
- We define a forward process that adds noise to an image and a the reverse process (ResUNet) that removes the noise to get the desired image. We define a loss function to allow the ResUNet to optimize upon: ELBO loss.

- The forward process:

- The reverse process:

### CondDDPM

- Same as DDPM but with condtionan labels being passed into the ResUNet. The labels are passed in as one-hot vectors and also there is a dropout variable omega to mask between contional predictions and non conditional predictions.

### ResUNet

- ResUNet is a UNet architecture with residual connections.

### Training Process:

- For diffusion models the forward pass is responsible for adding noise to an image based on a random time step t and attempt to reverse the noise to get the original image for a specific time step t. Sampling is done for inference in between epochs to check the quality of the generated images.
