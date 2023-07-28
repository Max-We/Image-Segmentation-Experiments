# 🔍 Image Segmentation Experiments 🔍

This projects implements a few algorithms and neural networks to detect objects in an image using OpenCV and YOLO.

The following commands are at your disposal:

```text
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  detect-aruco       Detect ArUco markers in an image.
  detect-coins       Detect coins in an image.
  detect-legos       Detect legos in an image.
  detect-legos-nn    Detect legos in an image using a custom trained YOLOv7 neural-net.
  detect-objects-nn  Detect objects in an image using the standard YOLOv3 neural net.
  detect-paper       Detect paper in an image and applies perspective transformation.
```

Example:

```shell
python3 main.py detect-paper ./images/good/good2.jpg
```

## Usage

Under the `./images` directory, some sample images are provided to perform image semgentation / object detection on.
The results of the commands are saved under `./output`

### detect-objects-nn

In order to use `detect-objects-nn`, please download the YOLOv3 model weights and place them under `./yolo/yolov3.weights`.

```shell
wget https://pjreddie.com/media/files/yolov3.weights
```
### detect-legos-nn

The model for the `detect-legos-nn` command has been custom trained on a dataset of lego bricks. It is deployed on Huggingface
and details about the training / performance can be found here:

- [Huggingface model card](https://huggingface.co/mw00/yolov7-lego)
- [Huggingface space (interactive demo)](https://huggingface.co/spaces/mw00/yolov7-lego)

Currently only a zero-shot model so the performance can be optimized a lot in future iterations of this project.
