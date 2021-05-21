#!/bin/env python

# Inputs:
#   4D fMRI file (nifti)
#   3D Binary brain mask that matches 4D fMRI dimensions (nifti)
#
# Process:
#   Detrends and regresses global signal from the fmri file
#
# Output:
#   Writes residualized nifti to directory where the user supplied fMRI
#   is located.

# Raymond Viviano
# August 10th, 2018

# Please email bugs to rayviviano@gmail.com


from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

# All other imports
import os, sys, traceback
import numpy as np
import nibabel as nib
from scipy.signal import detrend
from os.path import dirname, abspath, join


def demean(V):
    """
    Demeans data assuming that each row is the timecourse for a single voxel. 
    """    
    if len(V.shape) > 1:
        # Get the mean of each row
        V_mean = V.mean(axis=1)
        return V - V_mean[:, np.newaxis]
    else:
        # Vector with only one axis supplied
        return V - V.mean()


def global_signal_regression(subject_nii, mask_nii):
    """
    Calculates residuals of global signal regression for 
    every voxel in the brain of a subject. Writes a nifti file with 
    _gs_regress.nii.gz suffix.
    
    subject_nii : string
        Path to subject's processed fmri - e.g., output of ICA AROMA
    mask_nii : string
        Path to whole 3D brain mask of with compatible dimensions to fMRI 
        
    """
    # Load image to process
    fmri_img = nib.load(subject_nii)
    fmri_hdr = fmri_img.get_header()
    fmri_aff = fmri_img.get_affine()
    fmri_data = fmri_img.get_data()

    # Reshape 4d array as voxels by timepoints
    (x, y, z, t) = np.shape(fmri_data)
    vox_num = x * y * z
    vox_by_ts = fmri_data.reshape((vox_num, t), order="F")

    # Load brain mask
    mask_img = nib.load(mask_nii)
    mask_data = mask_img.get_data().astype(np.int)
    mask_vec = mask_data.reshape((vox_num, 1), order="F")

    # Create array that excludes non-brain voxels
    brain_voxels = np.delete(vox_by_ts, np.where(mask_vec == 0)[0], axis=0)

    # Calculate global signal
    gs = brain_voxels.mean(axis=0)  
    
    # Demean and detrend global signal Vector
    gs = demean(gs)
    gs = detrend(gs)
    
    # Constant for regression
    constant = np.ones((gs.shape[0], 1))
 
    # Regression matrix
    X = np.hstack((gs.reshape(gs.shape[0],-1), constant))
    
    # Check regression matrix for validity
    if np.isnan(X).any():
        raise ValueError('Regressor contains NaN')

    # Store mean of each voxel timeseries
    data_means = vox_by_ts.mean(axis=1)

    # Demean and Detrend Data
    vox_by_ts = demean(vox_by_ts)
    vox_by_ts = detrend(vox_by_ts, axis=1)

    # Regression
    Y = vox_by_ts.T 
    try:
        B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    except np.linalg.LinAlgError as e:
        traceback.print_exc(file=sys.stdout)
        raise Exception("Error: {0}".format(e))

    # Calculate Residual
    Y_res = Y - X.dot(B)

    # Transpose to get back to vox (rows) by time (cols)
    processed_vox = Y_res.T

    # Add voxel timeseries mean back to residual matrix
    data_means = data_means[:, np.newaxis]
    processed_vox += np.tile(data_means, (1, processed_vox.shape[1]))
    
    # Reshape data
    processed_data = processed_vox.reshape((x,y,z,t), order="F")

    # Save processed img
    proc_img = nib.Nifti1Image(processed_data, header=fmri_hdr,
                               affine=fmri_aff)

    proc_rel_path = (os.path.split(subject_nii)[1].split(".")[0] 
                     + "_gs_regress.nii.gz")
    proc_abs_path = join(dirname(subject_nii), proc_rel_path)
    nib.save(proc_img, proc_abs_path)
    

def main():
    usage = "python gs_regress.py <fMRI.nii.gz> <mask.nii.gz>"
    try:
        subject = abspath(sys.argv[1])
        mask = abspath(sys.argv[2])
        global_signal_regression(subject, mask)
    except:
        traceback.print_exc(file=sys.stdout)
        # Should be using getopts earlier in the function...
        print(usage)


if __name__ == "__main__":
    main()