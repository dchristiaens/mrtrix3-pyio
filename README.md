# mrtrix3-pyio

This repository provides a Python package to interface with MRtrix3 file formats.


## Installation

You can clone and `pip`-install the code with:

```
git clone https://github.com/dchristiaens/mrtrix3-pyio.git
cd mrtrix3-pyio
pip install .
```

## Example use

```
>>> from mrtrix3.io import load_mrtrix
>>> im = load_mrtrix('dwi.mif')      # read an image from file
>>> print(im.shape); print(im.vox)   # access propoperties
(96, 96, 50, 65)
(2.5, 2.5, 2.5, nan)
>>> im.data *= 2            # access and edit the image data as a numpy array
>>> im.save('dwi_x2.mif')   # save the result to a new file
```
