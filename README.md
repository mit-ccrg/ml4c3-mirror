# ml4c3
Machine Learning for Cardiology and Critical Care

## Contents
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Setup](#setup)
- [Run mode example](#run-mode-example)

## Documentation
You can find the doumentation related to this repo on our [Wiki](https://github.com/aguirre-lab/ml4c3/wiki).
There, information about data sources, flows and storage, as well as all modes, tools and scripts is presented.

### Other documentation
GE documentation is stored in a shared Partners Dropbox folder ([link](https://www.dropbox.com/sh/aocdkcw71ehdem1/AAB2ZX7JENEAaeDarZ_Y68Ila?dl=0)) and includes:
1. Physician's guide to the Marquette 12SL ECG analysis program
2. Guide to MuseDB search
3. Muse v9 XML developer's guide

## Contributing
If you are interested in working with ml4c3 group take a look at our workflow in the general aguirre-lab
[README](https://github.com/aguirre-lab/aguirre-lab/blob/master/README.md).

## Setup
1. Install [docker](https://docs.docker.com/get-docker/) and [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. Add user to group to run docker
    ```bash
    sudo usermod -aG docker $USER
    ```
3. Build docker image
    ```bash
    ./docker/build.sh [-c for CPU image]
    ```
4. Setup conda environment to install pre-commit into
    ```bash
    make setup
    ```
    > On macOS, you can install [`gmake`](https://formulae.brew.sh/formula/make) to call `setup`

5. Activate conda environment so that pre-commit hooks are run
    ```bash
    conda activate ml4c3
    ```

## Run mode example
Modes are run from the command line. Here is an example of how to call `tensorize_icu` mode:
```
./scripts/run.sh -c $PWD/ml4c3/recipes.py tensorize_icu \
--tensors /media/ml4c3/hd5_cabg \
--path_adt /media/ml4c3/cohorts_lists/adt_cabg>
--path_xref /media/ml4c3/cohorts_lists/xref_ca>
--adt_start_index 0 \
--adt_end_index 1000 \
--staging_dir ~/data/icu \
--staging_batch_size 200 \
--allow_one_source \
```

Information and arguments of all modes, tools and scripts available from our repo are described in
[Recipes](https://github.com/aguirre-lab/ml4c3/wiki/Recipes),
[Tools](https://github.com/aguirre-lab/ml4c3/wiki/Tools) and
[Scripts](https://github.com/aguirre-lab/ml4c3/wiki/Scripts), respectively.
