# ml4cvd
Machine Learning for CardioVascular Disease - MGH/MIT edition!

## Contents
- [Setup](#setup)
- [Modes](#modes)
- [Jupyer Lab](#jupyter-lab)
- [Run scripts](#run-scripts)
- [Tests](#tests)
- [TensorMaps](#tensormaps)
- [Tensorize ECGs](#tensorize-ecgs)
- [Organize XMLs](#organize-xmls)
- [Contribute](#contribute)

## Setup
1. install [docker](https://docs.docker.com/get-docker/) and [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. add user to group to run docker
    ```
    sudo usermod -aG docker $USER
    ```
3. build docker image
    ```
    ./docker/build.sh [-c for CPU image]
    ```
4. setup conda environment to install pre-commit into
    ```
    make setup
    ```
5. activate conda environment so that pre-commit hooks are run
    ```
    conda activate ml4cvd
    ```

## Modes

#### train
Dynamically generate, train, and evaluate a machine learning model.
```
./scripts/run.sh -t $PWD/ml4cvd/recipes.py \
--mode train \
--tensors /path/to/data \
--input_tensors ecg_signal \
--output_tensors patient_outcome \
--output_folder results \
--id my-experiment
```

#### infer
Evaluate model performance and save model predictions for inspection. The number of samples inferred is controlled by `--batch_size` and `--test_steps`
```
./scripts/run.sh -t $PWD/ml4cvd/recipes.py \
--mode infer \
--tensors /path/to/data \
--input_tensors ecg_signal \
--output_tensors patient_outcome \
--batch_size 64 \
--test_steps 100 \
--output_folder results \
--id my-inference
```

#### plot_ecg
Plot ECGs generated by the GE Muse system that have been tensorized to HD5 format. Supports plotting in two modes: `clinical` which plots the a composite view of the 12 leads that the GE Muse system would print and `full` which plots each lead individually for all 10 seconds.
```
./scripts/run.sh -c $PWD/ml4cvd/recipes.py \
--mode plot_ecg \
--plot_mode clinical \
--tensors /path/to/ecgs \
--output_folder results \
--id my-plots
```

### Deidentify data
Some compute resources may not be allowed to store Protected Health Information (PHI). Therefore we sometimes need to deidentify data before using those resources.

The script at [scripts/deidentify.py](../scripts/deidentify.py) currently supports deidentification of ECG HD5s and STS CSV files (including both feature & outcome spreadsheets, and bootstrap lists of MRNs). Deidenfication of additional data sources can be implemented using the modular approach documented in the script itself.

To deidentify ECG and STS data:
```bash
./scripts/run.sh -c -t \
    $PWD/scripts/deidentify.py \
    --starting_id 1 \
    --ecg_dir $HOME/data/ecg/mgh \
    --sts_dir $HOME/data/sts-data \
    --mrn_map $HOME/data/deid/mgh_mrn_deid_map.csv \
    --new_ecg_dir $HOME/data/deid/ecg/mgh \
    --new_sts_dir $HOME/data/deid/sts-data
```

## Jupyter Lab

> JupyterLab is a web-based interactive development environment for Jupyter notebooks, code, and data. [`source`](https://jupyter.org)

A Jupyter Lab instance can be run inside Docker containers with `ml4cvd` installed:

```
./scripts/run.sh -j [-p PORT, default is 8888]
```

If the notebook docker container is running locally, navigate to the link generated by the Jupyter server.

If the container is running remotely, you can either 1) connect to the notebook via the remote server address (e.g. `http://mithril:1234/?token=asdf`), or 2) map a local to the remote port using an ssh tunnel so you can navigate to `http://localhost:1234/?token=asdf`:

```
ssh -NL PORT:localhost:PORT USER@HOST
```

If changes to the code are made after a Jupyter Lab instance is launched, update the package within the Jupyter notebook by reinstalling and reimporting `ml4cvd`. The following code is run inside the notebook.
```
! pip install --user ~/ml
import ml4cvd
```

> replace `~/ml` with the path to the repo on your machine

## Run scripts
Run scripts with commonly used calls are stored in [this Dropbox folder](https://www.dropbox.com/sh/hjz7adj01x1erfs/AABnZifp1mUqs7Z_26zm4ly9a?dl=0).

### Script dispatcher

To distribute `train` calls across bootstraps, GPUs, and scripts, use [`scripts/dispatch.py`](https://github.com/aguirre-lab/ml/blob/er_dispatcher/scripts/dispatch.py):

```zsh
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
```
bash scripts/run.sh -T $PWD/tests
```

Some of the unit tests are slow due to creating, saving and loading `tensorflow` models.
To skip those tests to move quickly, run
```
./scripts/run.sh -T $PWD/tests -m '"not slow"'
```
Ensure you wrap `"not slow"` in single quotes.

pytest can also run specific tests using `::`. For example
```
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

```
./scripts/run.sh -c -t $PWD/scripts/make_tensor_maps_for_ecg_labels.py
```

The generated script is not initially formatted with `Black`. However, `pre-commit` should fix that prior to the script being committed to the repo.

## Tensorize ECGs

### Organize XMLs and Removing Duplicates
`ingest/ecg/organize_xmls.py` moves XML files from a single directory into the appropriate yyyy-mm directory.

`ingest/ecg/remove_xml_duplicates.py` finds and removes exact duplicate XML files, as defined by every bit of two files being identical, determined via SHA-256 hashing.

### Tensorize XMLs to HD5
`tensorize_partners` mode in `recipes.py` extracts data from all XML files and saves as [HD5 files](https://www.hdfgroup.org). Tensorization also removes duplicates that contain nearly the same information, except for minor differences, for example minor version changes in acquisition software. This duplicate detection is done by matching patient-date-time fields.

This mode is called with the following arguments:
`--xml_folder` to specify the directory containing ECG XMLs.
`--tensors` to specify the directory where tensorized HD5 files should be saved.

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

### Extract ECG metadata
`explore` mode in `recipes.py` extracts data specified by `--input_tensors` from all HD5 files given to `--tensors` and calculates summary statistics. Additionally, all metadata is saved to a large CSV file:

This CSV file will be used to construct a performant, queryable database to identify future cohorts for research projects.

### Other documentation
GE documentation is stored in a shared Partners Dropbox folder ([link](https://www.dropbox.com/sh/aocdkcw71ehdem1/AAB2ZX7JENEAaeDarZ_Y68Ila?dl=0)) and includes:
1. physician's guide to the Marquette 12SL ECG analysis program
2. guide to MuseDB search
3. muse v9 XML developer's guide

## Organize XMLs
After you extract XMLs from MuseVM, organize them into `yyyy-mm` directories:

```
./scripts/run.sh -c -t -m /media/2tb \
    $PWD/scripts/organize_xmls.py \
    --source_xml_folder /path/to/data \
    --destination_xml_folder /path/to/destination \
    --bad_xml_folder ~/bad_xmls \
    --method copy \
    --verbose
```
Be sure to mount necessary storage locations to Docker via `-m /location` after `run.sh`.

If a date cannot be obtained, the XML is considered bad, and will be moved to the `bad_xml_folder`.

You can copy the data, or move it.


## Contribute

### Issues
Every task has an issue, and each issue is labeled to help us stay organized.

New issues are created using one of our three [issue templates](https://github.com/aguirre-lab/ml/issues/new/choose): 1) new feature request or enhancement, 2) bug report, or 3) question.

We track issues and PRs on our [ECG project board](https://github.com/orgs/aguirre-lab/projects/3).

Good issues are clearly written and sufficiently small in scope to be addressed in one sprint of work (1-5 days).

If a new issue is low priority, it is added to the `To do (backlog)` column.

If a new issue is high priority, it is added to the `To do (current sprint)` column and addressed the current week.

Issues that are being actively worked on are moved to the `In progress (issues)` column.

Issues do not go in `In review (PRs)` column. Only PRs go there.

We prefer to close linked issues via PR instead of manually closing issues.

### Branches
Name your branch with your initials, a description of the purpose of the branch, and dashes between words:

```
git checkout -B er-fix-grid-ecg-plot
```

### Commit messages
We do not enforce strict commit message style, but try to follow good practices as described in this blog post: https://chris.beams.io/posts/git-commit/#capitalize.

### PRs
To contribute code or documentation changes from your branch to the `master` branch in the repo, open a PR.

Select `aguirre-lab/ml` instead of `broadinstitute/ml` (the core Broad repo) as the destination for your PR.

New PRs use our repo template by default. Describe the major changes, at a high level.

Assign at least one reviewer from Aguirre Lab.

Reviewers approve PRs before code is merged to `master`.

Reviewers review their assigned PRs within 48 hours. If your requested PR review is overdue, remind the reviewer on Slack.

When PRs are approved, all commits are "squash merged", e.g. combine all commits from the head branch into a single commit in the base branch. Also, the branch is automatically deleted.
