"""
File that contains the different processing steps, as well as the processor which handles the full
pre-processing pipeline.
"""

from abc import ABC
import subprocess
from pathlib import Path
import utils
import shutil
import math
import logging
from typing import Tuple
import pydicom
import SimpleITK as sitk
import numpy as np
from preprocessing.dataobject import (
    DataObject,
    DataObjectLoader,
    DataInputGenerator,
    DataObjectVisualizer,
    DicomPathInput,
    NiftiImgInput,
    NiftiObjectLoader,
)


class ProcessingStep(ABC):
    """
    Abstract processing step baseclass, contains general methods.

    Attributes:
    req_loader (DataObjectLoader, optional): The loader that is necessary to load the data.
    req_input_generator (DataInputGenerator): Necessary input generator to get the input data from
    the dataobject to use in the processing step.
    req_visualizer (DataObjectVisualizer, optional): Visualizer that visualizes the dataobject
    """

    req_loader: DataObjectLoader = None
    req_input_generator: DataInputGenerator = None
    req_visualizer: DataObjectVisualizer = None

    def __init__(self, step_name: str, step_cfg: dict) -> None:
        """
        Initializes the processing step and assigns a name and the configuration to use

        Args:
            step_name (str): Name of the processing step.
            step_cfg (dict): Configuration dictionary which contains the necessary information to
            run the processing step.
        """
        self.name = step_name
        self.cfg = step_cfg

    @staticmethod
    def _set_up_dirs(output_dir: Path) -> None:
        """Sets up the directories to output the dataobject to.

        Args:
            output_dir (Path): Output directory of the dataobject
        """
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    def get_name(self) -> str:
        """Returns the name of the processing step

        Returns:
            str: Name of the processing step
        """
        return self.name

    def get_handlers(
        self,
    ) -> Tuple[DataObjectLoader, DataInputGenerator, DataObjectVisualizer]:
        """Returns the handlers that are necessary to run the processing step.

        Returns:
            Tuple[DataObjectLoader, DataInputGenerator, DataObjectVisualizer]: Tuple containing the
            three necessary handlers.
        """
        return (self.req_loader, self.req_input_generator, self.req_visualizer)

    def process_data(self, data_object: DataObject) -> bool:
        """Process the data, this method is overwritten to contain the processing step and can be
        called with super() to do the basicc set-up for a processing step.

        Args:
            data_object (DataObject): dataobject that will be processed

        Returns:
            bool: returns True if the processing step was a succes, else False
        """
        data_object.set_handlers(self.get_handlers())

        load_data = self.cfg.get("load_data", False)
        if load_data:
            data_object.load_data()

        self._set_up_dirs(data_object.output_path)


class DicomToNifti(ProcessingStep):
    """
    Converts DICOM folder with seperate .dcm per slice to NIFTI format using dcm2niix. Assumes one
    scan per folder. Requires dcm2niix.exe added to PATH
    Input: DICOM folder
    Output: NIFTI file
    """

    req_input_generator: DataInputGenerator = DicomPathInput()

    def process_data(self, data_object: DataObject) -> bool:
        """Processes the data_object. Uses the dicom path to run dcm2niix. This converts the
        individual files in the dicom folder of the dataobject to a nifti file. If dcm2niix corrects
        for tilt or equalizes the scan, this corrected nifti is used for further processing.

        Args:
            data_object (DataObject): dataobject that should be processed

        Returns:
            bool: True if the step completed, else False
        """
        super().process_data(data_object)

        input_dicom_dir = data_object.get_input()
        curr_output_dir = data_object.output_path

        if self.cfg["save_metadata"]:
            first_dicom = list(input_dicom_dir.glob("*.dcm"))[0]
            meta_data = pydicom.read_file(first_dicom, stop_before_pixels=True)
            lines = [line + "\n" for line in meta_data.formatted_lines()]
            with open(curr_output_dir / "dcm_metadata.txt", "w") as f:
                f.writelines(lines)

        flag_list = [f"-{key} {value}" for key, value in self.cfg["flags"].items()]
        flag_string = " ".join(flag_list)
        command = f'dcm2niix -o "{str(curr_output_dir)}" {flag_string} "{str(input_dicom_dir)}"'
        logging.info(command)
        out_code = subprocess.run(command, shell=True, capture_output=True)
        if out_code.returncode != 0:
            logging.warning(
                f"Dicom to nifti conversion failed for {data_object.object_name}, not continuing pipeline"
            )
            return False

        output_paths = list(curr_output_dir.glob("*.nii"))

        if len(output_paths) < 1:
            return

        output_path = [
            output_path for output_path in output_paths if "ROI" not in str(output_path)
        ]
        output_path = output_path[0]
        for path in output_paths:
            if "_Tilt" in str(path) or "_Eq" in str(path):
                output_path = path
                logging.info(f'input "{input_dicom_dir}" was corrected for tilt')
                break

        data_object.set_path(output_path)

        log_dict = {"used_scan": str(output_path)}
        data_object.log_change("dcm2niix", log_dict)

        return True


class NiftiClamp(ProcessingStep):
    """
    Clamps the values below 'min' to be 'min' and above 'max' to be 'max'
    Input: NIFTI image
    Output: NIFTI image
    """

    req_loader: DataObjectLoader = NiftiObjectLoader()
    req_input_generator: DataInputGenerator = NiftiImgInput()

    def process_data(self, data_object: DataObject) -> bool:
        """
        Processes the nifti data of the dataobject to be clamped to the min and max values specified
        in the configuration file.

        Args:
            data_object (DataObject): dataobject that should be processed

        Returns:
            bool: True if the step completed, else False
        """
        super().process_data(data_object)
        nifti_img = data_object.get_input()

        log_dict = {}
        utils.log_nifti_info(log_dict, nifti_img, "input")

        transformed_img = sitk.Clamp(
            nifti_img, sitk.sitkUInt32, self.cfg["min"], self.cfg["max"]
        )

        if self.cfg["save_intermediate"]:
            file_name = f'{data_object.object_name}_{self.cfg["img_tag"]}.nii'
            output_path = data_object.output_path / file_name
            sitk.WriteImage(transformed_img, output_path)
            log_dict["output_path"] = str(output_path)
            data_object.set_path(output_path)

        utils.log_nifti_info(log_dict, transformed_img, "output")
        data_object.log_change("nifti_window", log_dict)
        data_object.data = transformed_img
        return True


class NiftiWindow(ProcessingStep):
    """
    Windows the values of the scan to be between 'min' and 'max', sets all other values to
    'outside_value'
    Input: NIFTI image
    Output: NIFTI image
    """

    req_loader: DataObjectLoader = NiftiObjectLoader()
    req_input_generator: DataInputGenerator = NiftiImgInput()

    def process_data(self, data_object: DataObject) -> bool:
        """
        Windows the nifti data in the data_object to be between min and max in the config file. The
        rest is set to outside_value of the config file.

        Args:
            data_object (DataObject): dataobject that should be processed

        Returns:
            bool: True if the step completed, else False
        """
        super().process_data(data_object)
        nifti_img = data_object.get_input()

        log_dict = {}
        utils.log_nifti_info(log_dict, nifti_img, "input")

        transformed_img = sitk.Threshold(
            nifti_img, self.cfg["min"], self.cfg["max"], self.cfg["outside_value"]
        )

        if self.cfg["save_intermediate"]:
            file_name = f'{data_object.object_name}_{self.cfg["img_tag"]}.nii'
            output_path = data_object.output_path / file_name
            sitk.WriteImage(transformed_img, output_path)
            log_dict["output_path"] = str(output_path)
            data_object.set_path(output_path)

        utils.log_nifti_info(log_dict, transformed_img, "output")
        data_object.log_change("nifti_window", log_dict)
        data_object.data = transformed_img
        return True


class RoughSubjectSelection(ProcessingStep):
    """
    Selects the skull based on the given thresholds, dilates this with the specified kernel size,
    then fills in the center. The input image is masked with the dilated filled skull. If 'crop' is
    True it crops the image with a margin to remove blank space around the subject.
    Input: Nifti image
    Output: Nifti image
    """

    req_loader: DataObjectLoader = NiftiObjectLoader()
    req_input_generator: DataInputGenerator = NiftiImgInput()

    def process_data(self, data_object: DataObject) -> bool:
        """
        Selects the skull based on the given thresholds, dilates this with the specified kernel
        size, then fills in the center. The input image is masked with the dilated filled skull. If
        'crop' is True it crops the image with a margin to remove blank space around the subject.

        Args:
            data_object (DataObject): dataobject that should be processed

        Returns:
            bool: True if the step completed, else False
        """
        super().process_data(data_object)
        nifti_img = data_object.get_input()

        log_dict = {}
        utils.log_nifti_info(log_dict, nifti_img, "input")

        bone_img = sitk.BinaryThreshold(
            nifti_img, self.cfg["low_thresh"], self.cfg["high_thresh"]
        )

        dilate_radius = self.cfg["dilate_radius"]
        dilate_kernel_size = (
            int(math.floor(dilate_radius / nifti_img.GetSpacing()[0])),
            int(math.floor(dilate_radius / nifti_img.GetSpacing()[1])),
            int(math.floor(dilate_radius / nifti_img.GetSpacing()[2])),
        )

        largest_cc = sitk.Threshold(
            sitk.RelabelComponent(sitk.ConnectedComponent(bone_img))
        )

        bone_dil = sitk.BinaryDilate(largest_cc, dilate_kernel_size)
        outside_bone = bone_dil
        for _slice in range(outside_bone.GetSize()[2]):
            curr_slice = outside_bone[:, :, _slice - 1]
            curr_slice[0, :], curr_slice[-1, :], curr_slice[:, 0], curr_slice[:, -1] = (
                0,
                0,
                0,
                0,
            )
            curr_slice = sitk.ConnectedThreshold(
                curr_slice, [(0, 0, 0)], lower=0, upper=0
            )
            outside_bone[:, :, _slice - 1] = sitk.Cast(curr_slice, sitk.sitkUInt32)
        inside_bone = sitk.InvertIntensity(outside_bone, 1)

        if self.cfg["crop"]:
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(inside_bone)
            _x, _y, _z, x_size, y_size, z_size = stats.GetBoundingBox(1)
            margin = self.cfg["margin"]
            x_start = max(0, _x - margin // 2)
            y_start = max(0, _y - margin // 2)
            z_start = max(0, _z - margin // 2)
            x_end = min(nifti_img.GetSize()[0], _x + x_size + margin // 2)
            y_end = min(nifti_img.GetSize()[1], _y + y_size + margin // 2)
            z_end = min(nifti_img.GetSize()[2], _z + z_size + margin // 2)
        else:
            x_start, y_start, z_start = 0, 0, 0
            x_end, y_end, z_end = nifti_img.GetSize()

        masked_img = sitk.Mask(nifti_img, inside_bone, -1000)
        cropped_img = masked_img[x_start:x_end, y_start:y_end, z_start:z_end]

        if self.cfg["save_intermediate"]:
            file_name = f'{data_object.object_name}_{self.cfg["img_tag"]}.nii'
            output_path = data_object.output_path / file_name

            sitk.WriteImage(cropped_img, output_path)
            log_dict["output_path"] = str(output_path)
            data_object.set_path(output_path)

        utils.log_nifti_info(log_dict, cropped_img, "output")
        data_object.log_change("rough_subject_selection", log_dict)
        data_object.data = cropped_img
        return True


class NiftiResample(ProcessingStep):
    """
    Resamples the Nifti image to a new space given the parameters in the configuration
    Input: Nifti image
    Outpu: Nifti image
    """

    req_loader: DataObjectLoader = NiftiObjectLoader()
    req_input_generator: DataInputGenerator = NiftiImgInput()

    interpolators: dict[str, int] = {"b_spline": sitk.sitkBSpline}

    def process_data(self, data_object: DataObject) -> bool:
        """
        Resamples the Nifti image to a new space given the parameters in the configuration

        Args:
            data_object (DataObject): dataobject that should be processed

        Returns:
            bool: True if the step completed, else False
        """

        super().process_data(data_object)
        nifti_img = data_object.get_input()

        log_dict = {}
        utils.log_nifti_info(log_dict, nifti_img, "input")
        resampled_size = [
            int(math.ceil(old_size * (old_space / new_space)))
            for old_size, old_space, new_space in zip(
                nifti_img.GetSize(), nifti_img.GetSpacing(), self.cfg["spacing"]
            )
        ]

        if not self.cfg.get("new_size", None):
            new_size = resampled_size
        else:
            new_size = self.cfg["new_size"]

        original_direction = nifti_img.GetDirection()
        nifti_img.SetDirection(
            np.identity(3).flatten()
        )  # Change direction to identity for transforms

        transform = sitk.Euler3DTransform()
        transform.SetCenter(
            nifti_img.TransformContinuousIndexToPhysicalPoint(
                np.array(nifti_img.GetSize()) / 2.0
            )
        )
        transform.SetTranslation(-(np.array(new_size) - np.array(resampled_size)) / 4)

        resampled_img = sitk.Resample(
            nifti_img,
            new_size,
            transform,
            self.interpolators[self.cfg["interpolator"]],
            nifti_img.GetOrigin(),
            self.cfg["spacing"],
            nifti_img.GetDirection(),
            defaultPixelValue=-1000,
        )

        resampled_img.SetDirection(
            original_direction
        )  # Change back to original direction

        if self.cfg["save_intermediate"]:
            file_name = f'{data_object.object_name}_{self.cfg["img_tag"]}.nii'
            output_path = data_object.output_path / file_name

            sitk.WriteImage(resampled_img, output_path)
            log_dict["output_path"] = str(output_path)
            data_object.set_path(output_path)

        utils.log_nifti_info(log_dict, resampled_img, "output")
        data_object.log_change("nifti_resample", log_dict)
        data_object.data = resampled_img
        return True


class Processor:
    """
    Contains the logic to go through the processing steps in the correct order. Interrups pipeline
    for object if a step fails.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initializes the processor using the configuration, creates the output folder and removes the
        existing files if 'clear_previous_output' = True in the configuration file.

        Args:
            cfg (dict): configuration file with all the configurations for the whole pre-processing
        """
        self.cfg = cfg
        self.processing_steps = []

        output_path = Path(cfg["dirs"]["output"])

        if cfg["dirs"]["clear_previous_output"] and output_path.exists():
            shutil.rmtree(output_path)
            output_path.mkdir()
        elif not output_path.exists():
            output_path.mkdir(parents=True)

    def add_step(self, processing_step: ProcessingStep) -> None:
        """
        Adds a step to the preprocessing pipeline.

        Args:
            processing_step (ProcessingStep): The processing step to be added.
        """
        self.processing_steps.append(processing_step)

    def run(self, data_object: DataObject) -> None:
        """
        Runs a dataobject throught the pipeline.

        Args:
            data_object (DataObject): The dataobject that is processed.
        """
        logging.basicConfig(level=logging.INFO)
        for step in self.processing_steps:
            logging.info(
                f"Running step {step.get_name()} for {data_object.object_name}"
            )
            succes = step.process_data(data_object)

            if not succes:
                break
