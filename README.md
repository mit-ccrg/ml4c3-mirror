# ml4cvd
Machine Learning for CardioVascular Disease - MGH/MIT edition!

## Contents
- [Setup](#setup)
- [Modes](#modes)
- [Jupyer Lab](#jupyter-lab)
- [Run scripts](#run-scripts)
- [Tests](#tests)
- [TensorMaps](#tensormaps)
- [Work with ECG XML files](#work-with-ecg-xml-files): remove duplicates, organize files, and tensorize to `.hd5`

## Setup
1. install [docker](https://docs.docker.com/get-docker/) and [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. add user to group to run docker
    ```bash
    sudo usermod -aG docker $USER
    ```
3. build docker image
    ```bash
    ./docker/build.sh [-c for CPU image]
    ```
4. setup conda environment to install pre-commit into
    ```bash
    make setup
    ```
5. activate conda environment so that pre-commit hooks are run
    ```bash
    conda activate ml4cvd
    ```

## Modes

Modes are either within `recipes.py` or a standalone script in `/scripts`. Either are run from Bash inside of a Docker container that has the necessary environment.

### `explore`
`explore` mode in `recipes.py` extracts data specified by `--input_tensors` from all HD5 files at `--tensors` and calculates summary statistics. The specified tensors are saved to a large CSV file, and histograms of continous tensors are generated for data in all time windows.

This mode also lets you cross-reference data in `--tensors` against data in `--reference_tensors` which must be a path to a `.csv` file (we do not yet support cross-referencing two sets of `.hd5` files against each other). The column on which the source and reference data are joined is specified via `--reference_join_tensors`; this argument accepts multiple values.

Analyses can be further substratified by a binary or categorical tensor specified via `--explore_stratify_label`.

```bash
./scripts/run.sh -c -t $PWD/ml4cvd/recipes.py \
--mode explore \
--tensors /storage/shared/ecg/mgh \
--sample_csv ~/dropbox/sts-data/sts-mgh.csv \
--source_name ecg \
--join_tensors ecg_patientid_clean_sts_newest \
--time_tensor  ecg_datetime_sts_newest \
--reference_tensors ~/dropbox/sts-data/sts-mgh.csv \
--reference_name sts \
--reference_join_tensors medrecn \
--input_tensors \
    ecg_rate_md_preop_newest \
    ecg_qrs_md_preop_newest \
    ecg_pr_md_preop_newest \
    ecg_qt_md_preop_newest \
    ecg_qtc_md_preop_newest \
    ecg_paxis_md_preop_newest \
    ecg_raxis_md_preop_newest \
    ecg_taxis_md_preop_newest \
    ecg_ponset_md_preop_newest \
    ecg_poffset_md_preop_newest \
    ecg_qonset_md_preop_newest \
    ecg_qoffset_md_preop_newest \
    ecg_toffset_md_preop_newest \
--output_folder ~/dropbox/sts-ecg \
--explore_save_output \
--id explore
```

You can further stratify summary statistics tables and histograms of continuous tensors by a categorical tensor map. Just specificy its name:

```bash
--explore_stratify_label sts_death \
```

Support for stratifying categorical tensors does not yet exist.

#### `train`
Train and evaluate a deep learning model with smart defaults.
```bash
./scripts/run.sh -t $PWD/ml4cvd/recipes.py \
--mode train \
--tensors /storage/shared/ecg/mgh \
--input_tensors ecg_signal \
--output_tensors patient_outcome \
--output_folder results \
--id my-experiment
```

#### `infer`
Evaluate model performance and save model predictions for inspection. The number of samples inferred is controlled by `--batch_size` and `--test_steps`
```bash
./scripts/run.sh -t $PWD/ml4cvd/recipes.py \
--mode infer \
--tensors /storage/shared/ecg \
--input_tensors ecg_signal \
--output_tensors patient_outcome \
--batch_size 64 \
--test_steps 100 \
--output_folder results \
--id my-inference
```

#### `plot_ecg`
Plot ECGs generated by the GE Muse system that have been tensorized to HD5 format. Supports plotting in two modes: `clinical` which plots the a composite view of the 12 leads that the GE Muse system would print and `full` which plots each lead individually for all 10 seconds.
```bash
./scripts/run.sh -c $PWD/ml4cvd/recipes.py \
--mode plot_ecg \
--plot_mode clinical \
--tensors /path/to/ecgs \
--output_folder results \
--id my-plots
```

### `deidentify`
Some compute resources may not be allowed to store Protected Health Information (PHI). Therefore we sometimes need to deidentify data before using those resources.

The script at [scripts/deidentify.py](../scripts/deidentify.py) currently supports deidentification of ECG HD5s and STS CSV files (including both feature & outcome spreadsheets, and bootstrap lists of MRNs). Deidenfication of additional data sources can be implemented using the modular approach documented in the script itself.

To deidentify data from different institutions, the `starting_id` for each institution should be far apart. For example, if deidentifying MGH and BWH ECGs for the first time, use `starting_id 1` for MGH and `5000001` for BWH.

To deidentify ECG and STS data for MGH for the first time:
```bash
./scripts/run.sh -c -t \
    $PWD/scripts/deidentify.py \
    --starting_id 1 \
    --mrn_map $HOME/dropbox/ecg/mgh-deid-map.csv \
    \
    --ecg_dir /storage/shared/ecg/mgh \
    --new_ecg_dir /storage/shared/ecg-deid/mgh \
    \
    --sts_dir $HOME/dropbox/sts-data \
    --new_sts_dir $HOME/dropbox/sts-data-deid
```

To deidentify ECG data for BWH for the first time:
```bash
./scripts/run.sh -c -t \
    $PWD/scripts/deidentify.py \
    --starting_id 50000001 \
    --mrn_map $HOME/dropbox/ecg/bwh-deid-map.csv \
    \
    --ecg_dir /storage/shared/ecg/bwh \
    --new_ecg_dir /storage/shared/deid/ecg/bwh
```

To incrementally de-identify data for an institution (e.g. to deidentify additional HD5 files without having to repeat the de-identification process), do not specificy `starting_id`. The pipeline uses the existing MRN mapping; `starting_id` is inferred from the existing data:
```bash
./scripts/run.sh -c -t \
    $PWD/scripts/deidentify.py \
    --mrn_map $HOME/dropbox/ecg/mgh-deid-map.csv \
    --ecg_dir /storage/shared/ecg/mgh \
    --new_ecg_dir /storage/shared/ecg-deid/mgh
```

If incrementally updating the mapping for an institution, `starting_id` can be explicitly set but take care it does not collide with any existing IDs.

## Jupyter Lab

> JupyterLab is a web-based interactive development environment for Jupyter notebooks, code, and data. [`source`](https://jupyter.org)

A Jupyter Lab instance can be run inside Docker containers with `ml4cvd` installed:

```bash
./scripts/run.sh -j [-p PORT, default is 8888]
```

If the notebook docker container is running locally, navigate to the link generated by the Jupyter server.

If the container is running remotely, you can either 1) connect to the notebook via the remote server address (e.g. `http://mithril:1234/?token=asdf`), or 2) map a local to the remote port using an ssh tunnel so you can navigate to `http://localhost:1234/?token=asdf`:

```bash
ssh -NL PORT:localhost:PORT USER@HOST
```

If changes to the code are made after a Jupyter Lab instance is launched, update the package within the Jupyter notebook by reinstalling and reimporting `ml4cvd`. The following code is run inside the notebook.
```bash
! pip install --user ~/ml
import ml4cvd
```

> replace `~/ml` with the path to the repo on your machine

## Run scripts
Run scripts with commonly used calls are stored in [this Dropbox folder](https://www.dropbox.com/sh/hjz7adj01x1erfs/AABnZifp1mUqs7Z_26zm4ly9a?dl=0).

### Script dispatcher

To distribute `train` calls across bootstraps, GPUs, and scripts, use [`scripts/dispatch.py`](https://github.com/aguirre-lab/ml/blob/er_dispatcher/scripts/dispatch.py):

```bash
python scripts/dispatch.py \
--gpus 0-3 \
--bootstraps 0-9 \
--scripts \
    ~/dropbox/ml4cvd_run_scripts/sts_ecg/train-simple.sh
    ~/dropbox/ml4cvd_run_scripts/sts_ecg/train-varied.sh
    ~/dropbox/ml4cvd_run_scripts/sts_ecg/train-deeper.sh
```

### Tests
To run unit tests in Docker:
```bash
bash scripts/run.sh -T $PWD/tests
```

Some of the unit tests are slow due to creating, saving and loading `tensorflow` models.
To skip those tests to move quickly, run
```bash
./scripts/run.sh -T $PWD/tests -m '"not slow"'
```
Ensure you wrap `"not slow"` in single quotes.

pytest can also run specific tests using `::`. For example
```bash
./scripts/run.sh -T $PWD/tests/test_recipes.py::TestRecipes::test_explore -m '"not slow"'
```

For more pytest usage information, checkout the [usage guide](https://docs.pytest.org/en/latest/usage.html).


## TensorMaps
A TensorMap takes any kind of data stored in an hd5 file, tags it with a semantic interpretation and converts it into a structured numpy tensor.

TensorMaps can be used as inputs, outputs, or hidden layers of models made by the model factory.

TensorMaps can perform any computation by providing a callback function called `tensor_from_file`.

A default `tensor_from_file` will be attempted when a callback `tensor_from_file` is not provided.

TensorMaps guarantee shape, name, interpretation and mapping from hd5 to numpy array.

### Build TMaps for labels from ECG reads
ECG reads contain valuable information that can be parsed into labels via TMaps. For example, one can train a model to classify an ECG as AFib or not by using a TMap that looks for matching phrases in the ECG read.

Our system involves a number of `c_$TASK.xlsx` files stored in Dropbox (under the `ecg/labeling` folder). We call these spreadsheets "label maps". They are simple for clinical collaborators (most of whom are nontechnical) to revise. The first column contains source phrases, e.g. `atrial fibrillation`. The second through N'th column contain the string of the label, e.g. `afib`.

The script `scripts/make_tensor_maps_for_ecg_labels.py` parses label maps into code, resulting in the creation of `ml4cvd/tensor_maps_ecg_labels.py`.

Whenever label maps are updated, re-run the generating script to update the TMaps; note the `-c` flag to use the CPU image, and the `-t` flag for interactive mode in case you need to place breakpoints and debug.

```bash
./scripts/run.sh -c -t $PWD/scripts/make_tensor_maps_for_ecg_labels.py
```

The generated script is not initially formatted with `Black`. However, `pre-commit` should fix that prior to the script being committed to the repo.

## Work with ECG XML files

### Remove duplicate XMLs
`scripts/remove_xml_duplicates.py` finds and removes exact duplicate XML files, as defined by every bit of two files being identical, determined via SHA-256 hashing.

```bash
./scripts/run.sh -c $PWD/scripts/remove_xml_duplicates.py \
--src /path/to/xmls-to-dedup
```

### Organize XMLs into `yyyy-mm` subdirectories
`scripts/organize_xmls.py` moves XML files from a single directory into the appropriate `yyyy-mm` directory:

```bash
./scripts/run.sh -c $PWD/scripts/organize_xmls.py \
--source_xml_folder /path/to/xmls-to-organize \
--destination_xml_folder /media/2tb/ecg_xmls \
--method copy \ # can be also be 'move'
--verbose
```

If a date cannot be obtainedi from the XML, that file is considered bad and it will be moved to `/bad_xml_folder`. This path can be set with `--bad_xml_folder PATH_YOU_SET`

### Tensorize XMLs to HD5
`tensorize` mode in `recipes.py` extracts ECG data from XML files and saves as [HD5 files](https://www.hdfgroup.org). Tensorization also removes duplicates that contain nearly the same information, except for minor differences, for example minor version changes in acquisition software. This duplicate detection is done by matching patient-date-time fields.

This mode is called with the following arguments:
`--xml_folder` to specify the directory containing ECG XMLs.
`--tensors` to specify the directory where tensorized HD5 files should be saved.

```bash
./scripts/run.sh -c $PWD/ml4cvd/recipes.py \
--mode tensorize \
--xml_folder /path/to/xml-files \
--tensors /path/to/hd5-files
```

All the ECGs belonging to one patient, identified by medical record number (MRN), will be saved to one HD5, indexed by ECG acquisition date and time:
```
<MRN>.hd5
└--partners_ecg_rest
   |--date_1
   |  └--ECG Data
   └--date_2
      └--ECG Data
```

### ECG data structure
TODO revise description of voltage
TODO compression adn decompression

Voltage is saved from XMLs as a dictionary of numpy arrays indexed by leads in the set `("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")`, e.g.:

```
voltage = {'I': array([0, -4, -2, ..., 7]),
          {'II': array([2, -9, 0, ..., 5]),
          ...
          {'V6': array([1, -4, -3, ..., 4]),
```

Every other element extracted from the XML is returned as a string, even if the underlying primitive type is a number (e.g. age).

### Other documentation
GE documentation is stored in a shared Partners Dropbox folder ([link](https://www.dropbox.com/sh/aocdkcw71ehdem1/AAB2ZX7JENEAaeDarZ_Y68Ila?dl=0)) and includes:
1. physician's guide to the Marquette 12SL ECG analysis program
2. guide to MuseDB search
3. muse v9 XML developer's guide
