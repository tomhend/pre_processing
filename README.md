# CT head preprocessor
## Important notice
When running on the AMC network, the pre-processing code occasionally just stops running. Be sure to check afterwards if all scans are processed. I have not found the reason for this.
## Introduction
This repository processes CT-scans of the head. The configurations are found in config_example, adapt these to your liking and save them as 'config.yaml' in the 'configs' folder. This is the configuration file the pipeline will use. Run the pipeline from terminal using the command 'python main.py' when having this repository as your current working directory. To run you require a .csv file that has a column named 'subjects' and one called 'scan_paths', which contains all subject names and file paths you wish to use. The utils file has some helper functions to create this file if necessary.

## Processing steps
### Dicom to nifti
The dicom_to_nifti step is unique in the sense that it's the only step that uses dicom folders as input to produce nifti files. For proper functioning of the further pipeline every folder should only contain the files of one CT-head. The dicoms are converted to nifti using [dcm2niix](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage), see the documentation to understand the caveats of this method. One important aspact of dcm2niix is the correction of slice thickness and tilt. This produces extra nifti outputs. Right now the corrected nifti is used for the rest of the pipeline.

### Rough subject mask
This method removes irrelevant objects outside of the patients head. It does so by determining the skull and dilating this by a radius provided in the configurations. Everything outside of this skull is removed. Issues can arise if bone density or higher objects are near the skull (specifically within 2x the dilating distance).

### Nifti window
Windows the nifti image to only keep values within the window and set the rest to the specified value.

### Nifti clamp
Clamps the values to minimum and maximum values

### Nifit resample
Resamples the image to a new space given the configuration parameters.