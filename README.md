# ca_diffusion
Calcium imaging video synthesis using Latent Flow Matching models.
This repo is currently under construction and mainly serves me for playing around and testing new stuff.

## What Is This Repository About?
Shortly summarized, this codebase transfers the idea of latent diffusion models (LDMs) to the domain of calcium imaging.
The main difference towards normal images/videos is the noise character of the data which is assumed to follow a Poisson-Gaussian distribution
and is thus dependent on the underlying clean structure itself. 
Vanilla application of VAE training as it is done in LDM will fail in the sense that noise will be averaged out through application of reconstruction losses.
This issue can be solved by having a probabilistic decoder as for instance diffusion autoencoders do.
To follow this approach this repository implements a convolution based diffusion autoencoder consisting of a causal convolutional encoder which injects a latent representation to a
causal 3D-UNet based diffusion model as condition to achieve reconstruction of the underlying tissue structure but with different noise.
Convolutional networks are choosen due to the small size of neurons which only require a small receptive field.
In the future I may also experiment with having a further global conditioning which is taken from a different view of the same video to disentangle structure and global noise and intensity parameters.
By training a second diffusion model on the latent representation space, generation of new underlying structures can be achieved (currently in progress).
The architectural design of the second stage is build up on diffusion transformers (DiTs).

Future plans involve control of neuron placment through semantic segemenation maps and experimenting with different learning strategies (for instance MAEs, metric learning, ...) on calcium imaging data.

## Features/ToDo 
- [x] Configure Lightning through Hydra
- [x] General dataloader for torch/webdataset/indexed webdatasets
- [ ] Adding support for FFCV, DeepLake
- [x] EMA
- [ ] Deepspeed EMA
- [x] Checkpointing 
- [x] Deepspeed checkpointing
- [ ] Try FSDP
- [ ] Profile to find best torch.compile placement
- [x] Flow Matching
- [ ] Continous DDPM
- [x] QK-Normalization
- [ ] RoPE
- [x] Causal 3D AE
- [ ] Magnitude preserving layers & architecture (EDM-2)
- [ ] Video logger
- [ ] Find a more elegant way to config first stage

## Installation
- Python 3.12 is recommended
- Install packages from requirements.txt

## Data
Currently two major calcium imaging datasets are supported: Neurofinder and Allen Brain Observatory.
- Run scripts in scripts/data to download & setup calcium imaging datasets

## Training
Run the following command for training or have a look at the exemplary SLURM script
```sh
python main.py experiment=<experiment name>
python main.py experiment=<experiment name> trainer.devices=[id1,id2]
```

## Results

### Autoencoding
Examplary sample from diffusion autoencoder trained for roughly 80k steps on neurofinder videos. Original videos have higher bit depth (.tiff) compared to RGB which may cause artifacts in the presented samples.
Autoencoder training is performed on 128x128x128 clips whereas inference can be run on higher resolutions (depending on GPU memory - or tile input). More samples with longer training will follow ...
(Left: GT mean, Middle left: GT, Middle right: AE prediction, Right: Prediction mean)
<br>
<img src="media/autoencoding_example.gif" width=512 height=128/>

### Unconditional Clip Sampling
A few unconditional examples drawn from the latent video transformer around 7k steps:
<br>
<img src="media/unconditional_sample.gif" width=776 height=260/>


### CA-Label Software
Along with this repository comes a simply calcium imaging labeling software which is currently still in development.
Basic recipe to label a calcium imaging video:
- Enter video path and press 'Load'
- Select the neighbouring pixels to use for correlation image calculation and press 'Pre-Process Data'
- It is recommended to save the pre-processed data for later in an .npz file
- If already a part of the video is annotated enter the path to the mask file (JSON) and press 'Load'
- Pick a mask by clicking on the main view and then untick 'Pick Mode' to add (left click) or remove (right click) pixels from the current mask. Maybe zoom in and use the 'Focus' button to get a better view.
- Alternatively, click 'Add' to create a new mask and then untick 'Pick Mode' and place the first pixel and then zoom in and focus for further labeling.
- Select a mode to choose to see neurons better. The slider next to it can be used to control the alpha level of the masks.
- To improve the contrast while labeling, insert lower and or upper intensity and then press 'Set'.
- If a mask is annotated press 'Measure Mask' to obtain the average signal. Use margin sliders to estimate background signal.
<br>
<img src="media/ca_label_screenshot.png" width=1024 height=1024/>

## Credits
This code base is inspired and partially borrows code and ideas from the following works. 
Please also have a look at their repositories and appreciate their wounderful work:
 - Hydra lightning integration: https://github.com/ashleve/lightning-hydra-template
 - Latent diffusion models: https://github.com/CompVis/latent-diffusion
 - k diffusion: https://github.com/crowsonkb/k-diffusion
