# Fork of [Krasjet-Yu/YOLO-FaceV2](https://github.com/Krasjet-Yu/YOLO-FaceV2)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.0. (🔥)
* Converted ONNX models from GitHub [releases page](https://github.com/clibdev/YOLO-FaceV2/releases). (🔥)
* The [wider_val.txt](data/widerface/val/wider_val.txt) file for WIDERFace evaluation. 
* The following deprecations and errors has been fixed:
  * UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
  * DeprecationWarning: 'np.float' is a deprecated alias for builtin 'float'.
  * FutureWarning: Cython directive 'language_level' not set.
  * Cython Warning: Using deprecated NumPy API.
  * AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

| Name        | Link                                                                                                                                                                                          |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| yolo-facev2 | [PyTorch](https://github.com/clibdev/YOLO-FaceV2/releases/latest/download/yolo-facev2_last.pt), [ONNX](https://github.com/clibdev/YOLO-FaceV2/releases/latest/download/yolo-facev2_last.onnx) |

# Inference

```shell
python detect.py --weights yolo-facev2_last.pt --source data/images/bus.jpg
```

# WIDERFace evaluation

* Download WIDERFace [validation dataset](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view).
* Move dataset to `data/widerface/val` directory.

```shell
python widerface_pred.py --weights yolo-facev2_last.pt --dataset_folder data/widerface/val/images
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
python models/export.py --weights yolo-facev2_last.pt --grid
```
