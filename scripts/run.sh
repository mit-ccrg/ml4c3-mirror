#!/usr/bin/env bash

# Script to enable running Python modules within Docker containers

################### VARIABLES ############################################

DOCKER_IMAGE_GPU="ghcr.io/aguirre-lab/ml4c3:latest-gpu"
DOCKER_IMAGE_NO_GPU="ghcr.io/aguirre-lab/ml4c3:latest-cpu"
DOCKER_IMAGE=${DOCKER_IMAGE_GPU}
GPU_DEVICE="--gpus all"
MOUNTS=""
PORT="8888"
PORT_FLAG=""
PYTHON_COMMAND="python"
TEST_COMMAND="python -m pytest"
JUPYTER_COMMAND="jupyter lab --allow-root"
SCRIPT_NAME=$( echo $0 | sed 's#.*/##g' )
CONTAINER_NAME=""

################### USERNAME & GROUPS ####################################

# Get group ids and names, interleaved: id1 group1 id2 group2 ...
GROUP_IDS_NAMES=$(id ${USER} | sed 's/.*groups=//g' | awk -F'[,()]' '{for (i=1; i<=NF; i++) {if ($i == "") {continue} print $i}}')

# Export environment variables so they can be passed into Docker and accessed in bash
export GROUP_IDS_NAMES

# Create string to be called in Docker's bash shell via eval;
# this creates a user, adds groups, adds user to groups, then calls the Python script
# the 'staff' group is a preloaded group in the docker image which enables access to root owned folders
SETUP_USER="
    useradd -u $(id -u) -d ${HOME} ${USER};
    GROUPS_ARR=( \${GROUP_IDS_NAMES} );
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
CALL_AS_USER="sudo SETUPTOOLS_USE_DISTUTILS=stdlib -H -u ${USER}"

################### HELP TEXT ############################################

usage()
{
    cat <<USAGE_MESSAGE

    This script can be used to run a Python module within a Docker container.

    Usage: ${SCRIPT_NAME} [-nth] [-i <image>] module [arg ...]

    Example: ./${SCRIPT_NAME} -n -t -i ml4c3:latest-cpu recipes.py --mode tensorize ...

        -c                  if set use CPU docker image and machine and use the regular 'docker' launcher.
                            By default, we assume the machine is GPU-enabled.

        -d                  Select a particular GPU device on multi GPU machines

        -m                  Directories to mount at the same path in the docker image

        -j                  Run Jupyer Lab

        -p                  Ports to map between docker container and host

        -r                  Call Python script as root. If this flag is not specified,
                            the owner and group of the output directory will be those
                            of the user who called the script.

        -h                  Print this help text.

        -i      <image>     Run Docker with the specified custom <image>. The default image is '${DOCKER_IMAGE}'.
        -T                  Run tests
USAGE_MESSAGE
}

################### OPTION PARSING #######################################

while getopts ":i:d:m:p:cjtrhT" opt ; do
    case ${opt} in
        h)
            usage
            exit 1
            ;;
        i)
            DOCKER_IMAGE=$OPTARG
            ;;
        d)
            GPU_DEVICE="--gpus device=${OPTARG}"
            ;;
        m)
            MOUNTS="${MOUNTS} -v ${OPTARG}:${OPTARG}"
            ;;
        j)
            PYTHON_COMMAND=${JUPYTER_COMMAND}
            INTERACTIVE="-it"
            ;;
        p)
            PORT="${OPTARG}"
            PORT_FLAG="-p ${PORT}:${PORT}"
            ;;
        c)
            DOCKER_IMAGE=${DOCKER_IMAGE_NO_GPU}
            GPU_DEVICE=
            ;;
        r) # Output owned by root
            SETUP_USER=""
            CALL_AS_USER="SETUPTOOLS_USE_DISTUTILS=stdlib"
            ;;
        T)
            PYTHON_COMMAND=${TEST_COMMAND}
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

if [[ $# -eq 0 && "$PYTHON_COMMAND" != "$JUPYTER_COMMAND" ]]; then
    echo "ERROR: No Python module was specified." 1>&2
    usage
    exit 1
fi

################### SCRIPT BODY ##########################################

if ! docker pull ${DOCKER_IMAGE}; then
    echo "Could not pull the image ${DOCKER_IMAGE}."
fi

if [[ -d "/storage" ]] ; then
    echo "Found /storage; will try to mount it."
    MOUNTS="${MOUNTS} -v /storage/:/storage/"
fi

if [[ -d "/media" ]] ; then
    echo "Found /media; will try to mount it."
    MOUNTS="${MOUNTS} -v /media/:/media/"
fi

PYTHON_ARGS="$@"
if [[ "$PYTHON_COMMAND" == "$JUPYTER_COMMAND" ]] ; then
    if [[ "${PORT_FLAG}" == "" ]] ; then
        PORT_FLAG="-p ${PORT}:${PORT}"
    fi
    FQDN=$(hostname -A | awk '{for(i=1;i<=NF;i++) print length($i), $i}' | sort -nr | head -1 | awk '{print $2}')
    PYTHON_ARGS="--port ${PORT} --ip 0.0.0.0 --no-browser --NotebookApp.custom_display_url http://${FQDN}:${PORT}"
    CONTAINER_NAME="--name ${USER}-notebook-${PORT}"
fi

PYTHON_PACKAGES="$PWD"
if [[ -d $PWD/../icu ]] ; then
    PYTHON_PACKAGES="$PYTHON_PACKAGES $PWD/../icu"
fi

cat <<LAUNCH_MESSAGE
Attempting to run Docker with
    docker run --rm -it \
    ${GPU_DEVICE} \
    --env GROUP_IDS_NAMES \
    --uts=host \
    --ipc=host \
    -v ${HOME}/:${HOME}/ \
    ${MOUNTS} \
    ${PORT_FLAG} \
    ${CONTAINER_NAME} \
    ${DOCKER_IMAGE} /bin/bash -c \
    "${SETUP_USER}
    cd ${HOME};
    ${CALL_AS_USER} pip install ${PYTHON_PACKAGES};
    ${CALL_AS_USER} ${PYTHON_COMMAND} ${PYTHON_ARGS}"
LAUNCH_MESSAGE

docker run --rm -it \
${GPU_DEVICE} \
--env GROUP_IDS_NAMES \
--uts=host \
--ipc=host \
-v ${HOME}/:${HOME}/ \
${MOUNTS} \
${PORT_FLAG} \
${CONTAINER_NAME} \
${DOCKER_IMAGE} /bin/bash -c \
"${SETUP_USER}
cd ${HOME};
${CALL_AS_USER} pip install ${PYTHON_PACKAGES};
${CALL_AS_USER} ${PYTHON_COMMAND} ${PYTHON_ARGS}"
