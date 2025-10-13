# ASTNet: Adaptive Spatio-temporal Transformer for Efficient Video Deblurring

<hr />

> **Abstract**: *Removing motion blur from videos is a highly challenging task, with existing methods primarily focused on extracting spatio-temporal information from blurred frames and effectively fusing it. Among these, methods based on Transformer have demonstrated particularly outstanding performance in video deblurring tasks. However, the standard self-attention mechanism considers all regions within the feature map, making it difficult to adapt to certain non-uniform blur scenarios. To address this, we propose an adaptive self-attention mechanism that uses the ReLU function and square operation to filter out negative values in attention scores, thereby enhancing the network's ability to focus on locally severely blurred regions. To accommodate global blurring scenarios, we retain the dense attention branch. To further enable fine-grained feature adjustment, we introduce a bidirectional feature enhancement feed-forward network, which can simultaneously extract spatio-temporal features in both horizontal and vertical directions. We also designed a dynamic feature fusion module to adaptively integrate feature information from neighboring frames to avoid error accumulation during feature fusion. Experimental results on synthetic and real datasets show that the proposed method outperforms the current state-of-the-art methods.*
<hr />

## Network Architecture

![Network](figs/Network.png)
![Sparse Transformer](figs/SparseTransformer.png)
![FFN and PDTFM](figs/FNN_PDTFM.png)

## Experimental Results
Quantitative evaluations on the GoPro dataset.
![GOPRO_PSNR_SSIM](figs/GOPRO_PSNR_SSIM.png)

Quantitative evaluations on the DVD dataset.
![DVD_PSNR_SSIM](figs/DVD_PSNR_SSIM.png)

Quantitative evaluations on the BSD dataset.
![BSD_PSNR_SSIM](figs/BSD_PSNR_SSIM.png)

Deblurring results on the GOPRO dataset.
![GOPRO_Result](figs/GOPRO_Result.png)

Deblurring results on the DVD dataset.
![DVD_Result](figs/DVD_Results.png)

Deblurring results on the BSD dataset.
![BSD_Result](figs/BSD_Results.png)

Deblurring results on the Real blurry video.
![Real_Result](figs/REAL_Results.png)

## Dependencies
- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.10.1](https://pytorch.org/): `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge`
- Install dependent packages :`pip install -r requirements.txt`
- Install DSTNet :`python setup.py develop`

## Get Started

### Pretrained models
- Models are available in  `'./experiments/model_name'`

### Dataset Organization Form
If you prepare your own dataset, please follow the following form like GOPRO/DVD:
```
|--dataset  
    |--blur  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--gt
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
        	:
        |--video n
```
 
### Training
- Download training dataset like above form.
- Run the following commands:
```
Single GPU
python basicsr/train.py -opt options/train/train_GOPRO.yml
Multi-GPUs
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train/train_GOPRO.yml --launcher pytorch
```

### Testing
- Models are available in  `'./experiments/'`.
- Organize your dataset(GOPRO/DVD/BSD) like the above form.
- Run the following commands:
```
python basicsr/test.py -opt options/test/test_Deblur_GOPRO.yml
cd results
python merge_full.py
python calculate_psnr.py
```
- Before running merge_full.py, you should change the parameters in this file of Line 5,6,7,8.
- The deblured result will be in `'./results/dataset_name/'`.
- Before running calculate_psnr.py, you should change the parameters in this file of Line 5,6.
- We calculate PSNRs/SSIMs by running calculate_psnr.py

## Citation
```

```
