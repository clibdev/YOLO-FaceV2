# Fork of [Krasjet-Yu/YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.0. (🔥)
* The [wider_val.txt](data/widerface/val/wider_val.txt) file for WIDERFace evaluation. 
* The following deprecations and errors has been fixed:
  * UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
  * AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'.

# Installation

```shell
pip install -r requirements.txt
```


# YOLO-FaceV2

## Step-Through Example
### Installation
Get the code.    
```shell
git clone https://github.com/Krasjet-Yu/YOLO-FaceV2.git
```

### Dataset
Download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset. Then convert it to YOLO format.
```shell
# You can modify convert.py and voc_label.py if needed.
python3 data/convert.py
python3 data/voc_label.py
```

## Preweight
The link is [yolo-facev2s.pt](https://github.com/Krasjet-Yu/YOLO-FaceV2/releases/download/v1.0/preweight.pt)


### Training
Train your model on WIDER FACE.
```shell
python train.py --weights preweight.pt    
                --data data/WIDER_FACE.yaml    
                --cfg models/yolov5s_v2_RFEM_MultiSEAM.yaml     
                --batch-size 32   
                --epochs 250
```

### Test
```shell
python detect.py --weights ./preweight/best.pt --source ./data/images/test.jpg --plot-label --view-img
```

### Evaluate    

Evaluate the trained model via next code on WIDER FACE   
        
If you don't want to train, you can also directly use our trained model to evaluate.   

The link is [yolo-facev2_last.pt](https://github.com/Krasjet-Yu/YOLO-FaceV2/releases/download/v1.0/best.pt)     


```shell
python widerface_pred.py --weights runs/train/x/weights/best.pt     
                         --save_folder ./widerface_evaluate/widerface_txt_x    
cd widerface_evaluate/    
python evaluation.py --pred ./widerface_txt_x
```
Download the *[eval_tool](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip)* to show the performance.    
    
The result is shown below:    

![](data/images/eval.png)


## Finetune
see in *[https://github.com/ultralytics/yolov5/issues/607](https://github.com/ultralytics/yolov5/issues/607)*
```shell
# Single-GPU
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve

# Multi-GPU
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > evolve_gpu_$i.log &
done

# Multi-GPU bash-while (not recommended)
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  "$(while true; do nohup python train.py... --device $i --evolve 1 > evolve_gpu_$i.log; done)" &
done
```
