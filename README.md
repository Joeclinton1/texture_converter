# Texture converter
Converts texture images to svg files for use in the Scratch project ["Stamped textured tri fill"](https://scratch.mit.edu/projects/723419988/)
## Installation

0. If you don't have python on your computer download the latest version here: https://www.python.org/downloads/

1. Download the source code to your computer by clicking the code button on the main page of this repository then clicking "Download Zip". Extract the zip file after downloading.

2. Install dependencies by running in cmd:
```
pip install lxml
pip install opencv-python
```

3. (opt) Add python to PATH environment variable by following this [tutorial](https://www.educative.io/answers/how-to-add-python-to-path-variable-in-windows)

## How to use
Read this first if you have never used python or command line: https://opensourceoptions.com/blog/how-to-run-python-scripts-from-the-command-line/

In the command line do:
1. Cd to the correct directory
2. Type ```python main.py -s <path to texture>```
3. The texture wil be outputted to the folder `/Out`

There are optional arguments to change the parameters explained below in the Arguments section


## Arguments
optional arguments:
- -h, --help:            show this help message and exit
- -d, --dim: Width and height in pixels of output image. def
-  -b, --bbsize: Size of bounding box in pixels
- -s, --source: Input textures separated by spaces
- -o, --out: Output directory
-  -m, --mode:  Mode to use for converting texture to subtextures. Can be 'SPLIT' or 'OVERLAP'
-  -D, --debug: Set to 1, to export a debug image

## Example usage

- Use default settings and default texture
```python main.py```
- Use default settings and provide own source texture
```python main.py -s "path/to/image.png"```
- Set all parameters to the default settings
```python main.py -s "path/to/image.png" -d 64 64 -b 2048 -o "path/to/output/dir" -m "SPLIT" -D 0```
