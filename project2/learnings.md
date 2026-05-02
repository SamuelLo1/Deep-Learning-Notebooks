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
