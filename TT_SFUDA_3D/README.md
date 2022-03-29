
## Requirement.

Download the [Dataset]() and [Source-trained Model Weights]() as mentioned previously. 

Please make sure dataset path and source-trained path are changed accordingly in "tt_sfuda_3d.py" script.

## Train the model.

In --cfg provide the corresponding experiment config file name (eg:unet_flair,unet_t1,unet_t2). 
```
CUDA_VISIBLE_DEVICES=1 python tt_sfuda_3d.py --cfg unet_flair 
```
