
## Requirement.

Download the [Dataset]() and [Source-trained Model Weights]() as mentioned previously. 

## Train the model.

In --source provide the source-trained model name (hrf_unet,rite_unet,chase_unet) and in --target provide to the target domain name (rite,hrf,chase). 
```
CUDA_VISIBLE_DEVICES=1 python tt_sfuda_2d.py --source hrf_unet --target rite
```
