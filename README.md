# ml4c3
Machine Learning for Cardiology and Critical Care

## Setup
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

## Documentation
See the [`ml4c3` wiki](https://github.com/aguirre-lab/ml4c3/wiki).

## Contributing
Read how to contribute on our [group's `README`](https://github.com/aguirre-lab/aguirre-lab#github).

Our team prioritizes in the following order:
1. Internal (our) PRs
1. Internal issues, especially if they block research
1. Externally submitted PRs
1. Externally submitted bug reports, feature requests, or questions

If you find an issue, please also submit a PR that solves the issue.
