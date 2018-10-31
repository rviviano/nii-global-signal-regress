Simple python script to detrend and regress global signal from 4D nifti files

Requires:
	- Mostly preprocessed nifti fmri (e.g., after motion correction and 
	  filtering, or output from ICA Aroma)
	- 3D Brain mask with compatible dimensions to the 4D nifti

Dependencies:
	- Numpy/SciPy
	- Nibabel