# ca_diffusion
Calcium imaging video synthesis using Latent Flow Matching models.
This repo is currently under construction and serves me for playing around with new stuff.

## Installation
- Python 3.12 is recommended
- Install packages from requirements.txt

## Data
- Run download scripts in scripts/data to setup calcium imaging datasets

## Training
Run the following command for training
```sh
python main.py experiment=<experiment name>
python main.py experiment=<experiment name> trainer.devices=[id1,id2]
```

## Results

### Autoencoding
Examplary sample from diffusion autoencoder trained for roughly 70k steps on neurofinder videos.
(Left: GT mean, Middle left: GT, Middle right: AE prediction, Right: Prediction mean)
<br>
<img src="media/autoencoding_example.gif" width=512 height=128/>

### Unsupervised Clip Sampling

## Credits
This code base is inspired and partially borrows code and ideas from the following works. 
Please also have a look at their repositories and appreciate their wounderful work:
 - Hydra lightning integration: https://github.com/ashleve/lightning-hydra-template
 - Latent diffusion models: https://github.com/CompVis/latent-diffusion
 - k diffusion: https://github.com/crowsonkb/k-diffusion
