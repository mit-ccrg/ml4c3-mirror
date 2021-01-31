#!/usr/bin/env bash

# Script to enable running Python modules within Docker containers

################### VARIABLES ############################################

DOCKER_IMAGE="ghcr.io/aguirre-lab/ml4c3:latest-cpu"
DOCKER_IMAGE_GPU="ghcr.io/aguirre-lab/ml4c3:latest-gpu"
GPU_DEVICE=
INTERACTIVE="-it"
MOUNTS=""
PORT="8888"
PORT_FLAG=""
PYTHON_COMMAND="python"
SHELL_COMMAND="bash"
TEST_COMMAND="python -m pytest"
JUPYTER_COMMAND="jupyter lab --allow-root"
VISUALIZER_COMMAND="python $PWD/ml4c3/recipes.py visualize --debug"
SCRIPT_NAME=$( echo $0 | sed 's#.*/##g' )
CONTAINER_NAME=""
ENV_VARS=""

################### USERNAME & GROUPS ####################################

# Get group ids and names, interleaved: id1 group1 id2 group2 ...
GROUP_IDS_NAMES=$(id ${USER} | sed 's/.*groups=//g' | awk -F'[,()]' '{for (i=1; i<=NF; i++) {if ($i == "") {continue} print $i}}')

# Create string to be called in Docker's bash shell via eval;
# this creates a user, adds groups, adds user to groups, then calls the Python script
# the 'staff' group is a preloaded group in the docker image which enables access to root owned folders
SETUP_USER="
    useradd -u $(id -u) -d ${HOME} ${USER};
    GROUPS_ARR=( ${GROUP_IDS_NAMES} );
    for (( i=0; i<\${#GROUPS_ARR[@]}; i=i+2 )); do
        echo \"Creating group\" \${GROUPS_ARR[i+1]} \"with gid\" \${GROUPS_ARR[i]};
        groupadd -f -g \${GROUPS_ARR[i]} \${GROUPS_ARR[i+1]};
        echo \"Adding user ${USER} to group\" \${GROUPS_ARR[i+1]}
        usermod -aG \${GROUPS_ARR[i+1]} ${USER}
    done;
    echo \"Adding user ${USER} to group staff\";
    usermod -aG staff ${USER};
    echo \"Adding user ${USER} to group sudo\";
    usermod -aG sudo ${USER};
    echo -e \"password\npassword\" | passwd ${USER};
"

ENV_VARS="$ENV_VARS TF_FORCE_GPU_ALLOW_GROWTH=true"
ENV_VARS="$ENV_VARS PYTHONNOUSERSITE=true"
ENV_VARS="$ENV_VARS SETUPTOOLS_USE_DISTUTILS=stdlib"
ENV_VARS="$ENV_VARS NUMEXPR_MAX_THREADS=8"

CALL_AS_USER="sudo $ENV_VARS -H -u ${USER}"

################### HELP TEXT ############################################

usage()
{
    cat << USAGE_MESSAGE

    This script can be used to run a Python module within a Docker container.

    Usage: ${SCRIPT_NAME} [-d <gpu_id>] [-m <directory>] [-i <image>] [-p <port>] [-j|-s|-t|-v] [-nrh] module [arg ...]

    Example: ./${SCRIPT_NAME} -n -i ml4c3:latest-cpu recipes.py --mode tensorize ...

        -d  <gpu_id>     Select a particular GPU device; first GPU is '0'.

        -m  <directory>  Directories to mount at the same path in the docker image.

        -i  <image>      Run Docker with the specified custom <image>. The default image is '${DOCKER_IMAGE}'.

        -p  <port>       Port to map between docker container and host.

        -j               Run Jupyter Lab.

        -s               Run bash shell.

        -t               Run tests.

        -v               Run visualizer.

        -n               Run Docker container non-interactively.

        -r               Call Python script as root. If this flag is not specified,
                         the owner and group of the output directory will be those
                         of the user who called the script.

        -h               Print this help text.
USAGE_MESSAGE
}

################### OPTION PARSING #######################################

while getopts ":i:d:m:p:jstvnrh" opt ; do
    case ${opt} in
        h)
            usage
            exit 1
            ;;
        d)
            DOCKER_IMAGE=${DOCKER_IMAGE_GPU}
            GPU_DEVICE="--gpus \"device=${OPTARG}\""
            ;;
        m)
            MOUNTS="${MOUNTS} -v ${OPTARG}:${OPTARG}"
            MOUNTED_DIR=${OPTARG}
            ;;
        i)
            DOCKER_IMAGE=$OPTARG
            ;;
        p)
            PORT="${OPTARG}"
            PORT_FLAG="-p ${PORT}:${PORT}"
            ;;
        j)
            PYTHON_COMMAND=${JUPYTER_COMMAND}
            ;;
        s)
            PYTHON_COMMAND=${SHELL_COMMAND}
            ;;
        t)
            PYTHON_COMMAND=${TEST_COMMAND}
            ;;
        v)
            PYTHON_COMMAND=${VISUALIZER_COMMAND}
            ;;
        n)
            INTERACTIVE=""
            ;;
        r) # Output owned by root
            SETUP_USER=""

            CALL_AS_USER="SETUPTOOLS_USE_DISTUTILS=stdlib"
            ;;
        :)
            echo "ERROR: Option -${OPTARG} requires an argument." 1>&2
            usage
            exit 1
            ;;
        *)
            echo "ERROR: Invalid option: -${OPTARG}" 1>&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -eq 0 && \
      $PYTHON_COMMAND != $SHELL_COMMAND && \
      $PYTHON_COMMAND != $JUPYTER_COMMAND && \
      $PYTHON_COMMAND != $VISUALIZER_COMMAND ]]; then
    echo "ERROR: No Python module was specified." 1>&2
    usage
    exit 1
fi

################### SCRIPT BODY ##########################################

# Try to pull the specified docker image;
# if the user specifies a custom docker image that is local but not on ghcr.io,
# it cannot be pulled and the message below will be displayed.
if ! docker pull ${DOCKER_IMAGE}; then
    echo "Could not pull the image ${DOCKER_IMAGE}."
fi

# Setup mounted directories
if [[ "$SETUP_USER" ]]; then
  MOUNTS="${MOUNTS} -v ${HOME}/:${HOME}/"
fi

if [[ -d "/storage" ]] ; then
    echo "Found /storage; will try to mount it."
    MOUNTS="${MOUNTS} -v /storage/:/storage/"
fi

if [[ -d "/media" ]] ; then
    echo "Found /media; will try to mount it."
    MOUNTS="${MOUNTS} -v /media/:/media/"
fi

# Setup python command and args
PYTHON_ARGS="$@"

if [[ "$PYTHON_COMMAND" == "$JUPYTER_COMMAND" ]] ; then
    if [[ "${PORT_FLAG}" == "" ]] ; then
        PORT_FLAG="-p ${PORT}:${PORT}"
    fi
    FQDN=$(hostname -A | awk '{for(i=1;i<=NF;i++) print length($i), $i}' | sort -nr | head -1 | awk '{print $2}')
    PYTHON_ARGS="--port ${PORT} --ip 0.0.0.0 --no-browser --NotebookApp.custom_display_url http://${FQDN}:${PORT}"
    CONTAINER_NAME="--name ${USER}-notebook-${PORT}"
fi

if [[ "$PYTHON_COMMAND" == "$VISUALIZER_COMMAND" ]] ; then
    if [[ "${PORT_FLAG}" == "" ]] ; then
        PORT_FLAG="-p ${PORT}:${PORT}"
    fi
    PYTHON_ARGS="${PYTHON_ARGS} --port ${PORT} --address 0.0.0.0 --tensors ${MOUNTED_DIR}"
    CONTAINER_NAME="--name ${USER}-visualizer-${PORT}"
fi

# Setup heredoc
if [[ "$GPU_DEVICE" ]]
then
    GPU_DEVICE_STRING="$GPU_DEVICE "
else
    GPU_DEVICE_STRING="# no GPU device specified"
fi

if [[ "$MOUNTS" ]]
then
    MOUNTS_STRING="$MOUNTS"
else
    MOUNTS_STRING="# no mount points specified"
fi

if [[ "$PORT_FLAG" ]]
then
    PORT_FLAG="$PORT_FLAG"
else
    PORT_FLAG_STRING="# no ports specified"
fi

if [[ "$CONTAINER_NAME" ]]
then
    CONTAINER_NAME_STRING="$CONTAINER_NAME"
else
    CONTAINER_NAME_STRING="# container name not specified"
fi

cat << EOF

Attempting to run Docker with:
    docker run --rm \\
    ${INTERACTIVE} \\
    ${GPU_DEVICE_STRING} \\
    --uts=host \\
    --ipc=host \\
    ${MOUNTS_STRING} \\
    ${PORT_FLAG_STRING} \\
    ${CONTAINER_NAME_STRING} \\
    ${DOCKER_IMAGE} /bin/bash -c \\
    "${SETUP_USER}
    cd ${HOME};
    ${CALL_AS_USER} pip install $PWD;
    ${CALL_AS_USER} ${PYTHON_COMMAND} ${PYTHON_ARGS}"

EOF

# Run Docker
echo "=================== Running Docker ==================="

docker run --rm \
${INTERACTIVE} \
${GPU_DEVICE} \
--uts=host \
--ipc=host \
${MOUNTS} \
${PORT_FLAG} \
${CONTAINER_NAME} \
${DOCKER_IMAGE} /bin/bash -c \
"${SETUP_USER}
cd ${HOME};
${CALL_AS_USER} pip install $PWD;
${CALL_AS_USER} ${PYTHON_COMMAND} ${PYTHON_ARGS}"
