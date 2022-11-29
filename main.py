"""
File containing the script that is responsible for creating the pipeline and running the desired
dataobjects throught it.
"""

import concurrent.futures
import logging
from itertools import repeat
from pathlib import Path

import pandas as pd

import utils
from preprocessing.dataobject import DataObject
from preprocessing.processors import (
    DicomToNifti,
    NiftiClamp,
    NiftiResample,
    NiftiWindow,
    Processor,
    RoughSubjectSelection,
)

STEPS_DICTIONARY = {
    "dcm_to_nifti": DicomToNifti,
    "nifti_window": NiftiWindow,
    "rough_subject_mask": RoughSubjectSelection,
    "nifti_clamp": NiftiClamp,
    "nifti_resample": NiftiResample,
}


@utils.time_func
def main():
    """
    Function responsible for loading the configuration file, loading the input_csv file, creating
    the processor and steps and running the processor. Also handles multi-threading.

    Raises:
        KeyError: If step_name is not known in the STEP_DICTIONARY this raises a KeyError
    """
    logging.basicConfig(level=logging.INFO)

    cfg = utils.parse_cfg(Path("./configs/config.yaml"))
    input_df = pd.read_csv(cfg["dirs"]["input_csv"])
    processor = Processor(cfg)

    for step_name in cfg["step_list"]:
        try:
            processing_class = STEPS_DICTIONARY[step_name]
        except:
            raise KeyError(f"Step '{step_name}' not implemented")
        processing_step = processing_class(step_name, cfg[step_name])
        processor.add_step(processing_step)

    subject_subset = cfg["dirs"].get("subjects", None)

    # Code for running only the not processed files
    #processed_path = Path(r'path to processed')
    #subject_subset = [int(path.name) for path in processed_path.iterdir()]
    
    if subject_subset:
        input_df = input_df.loc[~input_df["subjects"].isin(subject_subset)]
    subject_list = input_df["subjects"]
    path_list = input_df["scan_paths"].apply(Path)

    if cfg["multi_processor"]:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cfg.get("max_workers", None)
        ) as executor:
            executor.map(
                run_processor,
                repeat(processor),
                subject_list,
                path_list,
                repeat(Path(cfg["dirs"]["output"])),
            )
        return

    for subject, input_path in zip(subject_list, path_list):
        run_processor(processor, subject, input_path, Path(cfg["dirs"]["output"]))


def run_processor(
    processor: Processor, input_subject: str, input_path: Path, output_path: Path
) -> None:
    """
    Function that runs the processor for a dataobject, given a processor, input_subject, input_path,
    and output_path. First creates the dataobject, than passes it to the processor.

    Args:
        processor (Processor): Processor that should be used.
        input_subject (str): Name of the dataobject (i.e. patient_id).
        input_path (Path): Path of the dataobject's data.
        output_path (Path): Path where processed files should be saved.
    """
    object_name = input_path.name
    if not input_path.is_dir():
        object_name = input_path.parent.name
    data_object = DataObject(
        str(input_subject), str(object_name), input_path, output_path
    )
    processor.run(data_object)


if __name__ == "__main__":
    main()
