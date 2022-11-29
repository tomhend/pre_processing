"""
File contains the classes related to dataobjects. Datahandlers: DataObjectvisualizer,
DataObjectLoader, DataInputGenerator, and their subclasses. DataObject itself is the class that
holds information and data necessary for the pre-processing pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import SimpleITK as sitk
import yaml


class DataObjectVisualizer(ABC):
    """
    Abstract base class for dataobject visualizer, not used currently
    """

    @abstractmethod
    def show_data(self, data_object: "DataObject") -> None:
        """
        Abstract method that has to implement a way to show the data in the subclass.

        Args:
            data_object (DataObject): The dataobject that needs to be visualized
        """
        pass


class DicomObjectVisualizer(DataObjectVisualizer):
    """
    Class that implements the visualizer for Dicom Objects, not currently implemented.
    """

    def show_data(self, data_object: "DataObject") -> None:
        """
        Method that shows the data, not currently implemented.
        Args:
            data_object (DataObject): The dataobject that needs to be visualized
        """
        pass


class NiftiObjectVisualizer(DataObjectVisualizer):
    """
    Class that implements the visualizer for Nifti Objects, not currently implemented.
    """

    def show_data(self, data_object: "DataObject") -> None:
        """
        Method that shows the data, not currently implemented.
        Args:
            data_object (DataObject): The dataobject that needs to be visualized
        """
        pass


class DataObjectLoader(ABC):
    """
    Abstract base class for dataobject loader
    """

    @abstractmethod
    def load_data(self, data_path: Path) -> Any:
        """
        Abstract method that has to implement a way to load the data in the subclass.

        Args:
            data_path (Path): Path to the file that has to be loaded
        """
        pass


class DicomObjectLoader(DataObjectLoader):
    def load_data(self, data_path: Path) -> None:
        """
        Dataloader for dicom objects, not currently implemented.

        Args:
            data_path (Path): Path to the dicom folder
        """
        return None


class NiftiObjectLoader(DataObjectLoader):
    def load_data(self, data_path: Path) -> sitk.Image:
        """
        Dataloader for nifti objects

        Args:
            data_path (Path): Path to the nifti image

        Returns:
            sitk.Image: the sitk.Image of the nifti file.
        """
        return sitk.ReadImage(str(data_path))


class DataInputGenerator(ABC):
    """
    Abstract base class for dataobject input generator
    """

    @abstractmethod
    def get_input(self, data_object: "DataObject") -> None:
        """
        In the subclass this method should return a input for a processing step.

        Args:
            data_object (DataObject): dataobject for which the input should be returned.
        """
        pass


class DicomPathInput(DataInputGenerator):
    """
    Class for generating processing step input for dicom objects
    """

    def get_input(self, data_object: "DataObject") -> Path:
        """
        Returns the path to the dicom_folder which has the data of the data_object.

        Args:
            data_object (DataObject): The data object for which the input should be generated

        Returns:
            Path: Path to the dicom folder
        """
        return data_object.data_path


class NiftiImgInput(DataInputGenerator):
    """
    Class for generating processing step input for nifti objects
    """

    def get_input(self, data_object: "DataObject") -> sitk.Image:
        """
        Returns the sitk.Image data of the data_object to be used in processing steps.

        Args:
            data_object (DataObject): The data object for which the input should be generated

        Returns:
            sitk.Image: sitk Image entity that containts the data of the nifti.
        """
        return data_object.data


@dataclass
class DataObject:
    """
    Dataobject class that holds the information necessary to progress through the processing steps
    and the data to the image.

    Attributes:
        root_name (str): Name of the set this dataobject belongs to (i.e. patient)
        object_name (str): Name of the object (i.e. name of the scan series)
        data_path (Path): Path to the data
        output_root (Path): Path to the root output folder
        loader (DataObjectLoader, optional): Loader that handles the loading of the data from path
        to data object.
        input_generator (DataInputGenerator, optional): Generator that retrieves the data from the
        data object to be used in a processing step.
        visualizer (DataObjectVisualizer, optional): Visualizer that is used to show the data.
        data (object):  Object that holds the data.
    """

    root_name: str
    object_name: str
    data_path: Path
    output_root: Path
    loader: DataObjectLoader = field(init=False)
    input_generator: DataInputGenerator = field(init=False)
    visualizer: DataObjectVisualizer = field(init=False)
    data: object = field(init=False)

    def __post_init__(self):
        """
        Method that is run after initializing the data_object. Creates the output path for this
        dataobject.
        """
        self.output_path = self.output_root / self.root_name / self.object_name
        if self.root_name == self.object_name:
            self.output_path = self.output_path.parent

    def load_data(self) -> None:
        """
        Uses the loader to load the data into the data object.
        """
        self.data = self.loader.load_data(self.data_path)

    def get_input(self) -> Any:
        """
        Uses the input_generator to return the input necessary for the processing step.

        Returns:
            Any: The object that should be used for the processing step.
        """
        return self.input_generator.get_input(self)

    def show_data(self, *args, **kwargs) -> None:
        """
        Visualizes the data, currently not used.
        """
        self.visualizer.show_data(self, *args, **kwargs)

    def set_handlers(
        self,
        handlers: Tuple[DataObjectLoader, DataInputGenerator, DataObjectVisualizer],
    ) -> None:
        """
        Sets the handlers of the dataobjects to the supplied handlers.

        Args:
            handlers (Tuple[DataObjectLoader, DataInputGenerator, DataObjectVisualizer]): Tuple
            containing the three datahandlers that should be used.
        """
        self.loader, self.input_generator, self.visualizer = handlers

    def set_path(self, new_path: Path) -> None:
        """Changes the path where the data can be found

        Args:
            new_path (Path): The new path where the data is stored.
        """
        self.data_path = new_path

    def log_change(self, name: str, log_dict: dict) -> None:
        """
        Logs a change made to the dataobject by writing it to the yaml file.

        Args:
            name (str): Name of the change
            log_dict (dict): Dictionary containing the information about the change
        """
        log_path = self.output_path / "change_log.yaml"

        if not log_path.exists():
            with open(log_path, "w") as f:
                log_dict = {name: log_dict, "last_step": name}
                yaml.safe_dump(log_dict, f, default_flow_style=False)
            return

        with open(log_path, "r") as f:
            log_yaml = yaml.safe_load(f)
            log_yaml.update({name: log_dict, "last_step": name})

        with open(log_path, "w") as f:
            yaml.safe_dump(log_yaml, f, default_flow_style=False)
