
## Requirement.

Download the [Dataset]() and [Source-trained Model Weights]() as mentioned previously. 

Please make sure dataset path and source-trained path are changed accordingly in "tt_sfuda_2d.py" script.

## Train the model.

In --source provide the source-trained model name (hrf_unet,rite_unet,chase_unet) and in --target provide to the target domain name (rite,hrf,chase). 
```
CUDA_VISIBLE_DEVICES=1 python tt_sfuda_2d.py --source hrf_unet --target rite
```
