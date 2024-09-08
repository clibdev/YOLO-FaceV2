# Fork of [Krasjet-Yu/YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.4. (🔥)
* Converted ONNX models from GitHub [releases page](https://github.com/clibdev/YOLO-FaceV2/releases). (🔥)
* The [wider_val.txt](data/widerface/val/wider_val.txt) file for WIDERFace evaluation. 
* The following deprecations and errors has been fixed:
  * UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
  * DeprecationWarning: 'np.float' is a deprecated alias for builtin 'float'.
  * FutureWarning: You are using 'torch.load' with 'weights_only=False'.
  * FutureWarning: Cython directive 'language_level' not set.
  * Cython Warning: Using deprecated NumPy API.
  * AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

* Download links:

| Name        | Model Size (MB) | Link                                                                                                                                                                                  | SHA-256                                                                                                                              |
|-------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| YOLO-FaceV2 | 36.2<br>71.8    | [PyTorch](https://github.com/clibdev/YOLO-FaceV2/releases/latest/download/yolo-facev2.pt)<br>[ONNX](https://github.com/clibdev/YOLO-FaceV2/releases/latest/download/yolo-facev2.onnx) | f424fd437b22711207c48aac4c205d844eaea2f3c12e2f3f7ecd7f9650634e44<br>8ad769e4e9fd5869baa4243367e041df901a228e121a1602deec85074d2bd7ed |

# Inference

```shell
python detect.py --weights yolo-facev2.pt --source data/images/bus.jpg
```

# WIDERFace evaluation

* Download WIDERFace [validation dataset](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view).
* Move dataset to `data/widerface/val` directory.

```shell
python widerface_pred.py --weights yolo-facev2.pt --dataset_folder data/widerface/val/images
```
```shell
cd widerface_evaluate
```
```shell
python setup.py build_ext --inplace
```
```shell
python evaluation.py
```

# Export to ONNX format

```shell
pip install onnx
```
```shell
python models/export.py --weights yolo-facev2.pt --grid
```
