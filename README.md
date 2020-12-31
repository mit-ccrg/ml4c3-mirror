## What is `ml4c3`?
Machine Learning for Cardiology and Critical Care (`ml4c3`) is a pipeline for  
working with complex physiological data. It does several key things:

- *Ingest* raw data into a pre-processed format.
- *Tensorize* ingested data into standard `hd5` or CSV formats compatible with the pipeline.
- *Map* desired data from `hd5` or CSV files to code via the `TensorMap` abstraction.
- *Visualize* ECGs, ICU waveforms, labs, and modeling results.
- *Explore* summary statistics and trends.
- *Train* supervised and reinforcement learning models that are constructed from  
    powerful, simple, and expressive command line arguments and `TensorMaps`.
- *Cluster* data to reveal patterns via unsupervised learning.

## How do I install `ml4c3`?
1. Install [`docker`](https://docs.docker.com/get-docker/) and [`anaconda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

1. Add user to group to run docker
    ```bash
    sudo usermod -aG docker $USER
    ```

1. Build GPU docker image:
    ```bash
    ./docker/build.sh
    ```

    and also the CPU-only docker image:
    ```bash
    ./docker/build.sh -c
    ```

1. Setup conda environment for `pre-commit`:
    ```bash
    make setup
    ```
    > On macOS, you can install [`gmake`](https://formulae.brew.sh/formula/make) to call `gmake setup`

1. Activate conda environment so that `pre-commit` hooks run:
    ```bash
    conda activate ml4c3
    ```

## Why does `ml4c3` exist?
When a researcher joins an academic research group, they get access to a workstation or
server and some data. Sometimes a more senior researcher shares some code via email or
Dropbox -- usually a long `main.py` script with now-outdated dependencies. Nothing is
documented. A manuscript published using the code does not link to an open-source GitHub
repo, nor does it sufficiently document the steps and settings needed to reproduce the
results. The paper merely includes the familar phrase "code is available upon request".

The researcher spends their first week trying to set up their compute environment so they
can simply run their predecessor's pipeline. The packages need updating, and a library was
compiled in a special way, but nobody knows exactly how. The next two weeks are spent
understanding the code. Eventually, the researcher refactors the entire pipeline. A month
later, they start training models.

This story is too common in academia. We think there is a better way, so we built `ml4c3` to:
- address limitations of Jupyter notebooks, one-off scripts, and glue-code.
- enhance collaborative workflow between researchers, following best practices.
- increase efficiency via excellent documentation and modular code.

## How do I use `ml4c3`?
See the [`ml4c3` wiki](https://github.com/aguirre-lab/ml4c3/wiki).

## Can I contribute?
Yes, we would love your help! See our lab [`CONTRIBUTING`](https://github.com/aguirre-lab/aguirre-lab/blob/master/CONTRIBUTING.md)
to learn how to open an issue or submit a PR.

We prefer PRs that solve issues, but a well-written issue that describes a problem and
its root cause is also helpful.

## Who built `ml4c3`?
`ml4c3` was built by the [Aguirre Lab](https://csb.mgh.harvard.edu/aaron_aguirre) at the
Center for Systems Biology, Wellman Center for Photomedicine, and Cardiovascular Research
Center at the Massachusetts General Hospital.

We are a team of students, postdoctoral researchers, and engineers who work on overlapping
research projects at the intersection of machine learning, computer science, cardiology,
and critical care.

Since we share research needs, computational tooling, and data, the benefits of closely
collaborating around a shared infrastructure were obvious.
