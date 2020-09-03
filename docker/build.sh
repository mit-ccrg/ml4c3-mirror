#!/usr/bin/env bash

# Build and tag a Docker image and tag it 'latest_<gpu|cpu>'.

# Stop the execution if any of the commands fails
set -e

################### VARIABLES ############################################

REPO="ghcr.io/aguirre-lab/ml"
CONTEXT="docker/"
CPU_ONLY="false"

BASE_IMAGE_GPU="tensorflow/tensorflow:2.1.0-gpu-py3"
BASE_IMAGE_CPU="tensorflow/tensorflow:2.1.0-py3"

LATEST_TAG_GPU="latest-gpu"
LATEST_TAG_CPU="latest-cpu"

SCRIPT_NAME=$( echo $0 | sed 's#.*/##g' )

RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No colour

################### HELPER FUNCTIONS ############################################

usage()
{
    cat <<USAGE_MESSAGE

    This script can be used to build and tag a 'ml4cvd' image and to tag the image as 'latest_<gpu|cpu>'.

    Usage: ${SCRIPT_NAME} [-d <path>] [-t <tag>] [-c] [-h]

    Example: ./${SCRIPT_NAME} -d /home/username/ml/ml4cvd/docker -cp

        -d      <path>      Path to directory where Dockerfile is located. Default: '${CONTEXT}'

        -c                  Build off of the cpu-only base image and tag image also as '${LATEST_TAG_CPU}'.
                            Default: Build image to run on GPU-enabled machines and tag image also as '${LATEST_TAG_GPU}'.

        -h                  Print this help text

USAGE_MESSAGE
}

################### OPTION PARSING #######################################

while getopts ":d:t:chp" opt ; do
    case ${opt} in
        h)
            usage
            exit 1
            ;;
        d)
            CONTEXT=$OPTARG
            ;;
        c)
            CPU_ONLY="true"
            ;;
		p)
			PUSH_TO_GHCR="true"
			;;
        :)
            echo -e "${RED}ERROR: Option -${OPTARG} requires an argument.${NC}" 1>&2
            usage
            exit 1
            ;;
        *)
            echo -e  "${RED}ERROR: Invalid option: -${OPTARG}${NC}" 1>&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

################### SCRIPT BODY ##########################################

if [[ ${CPU_ONLY} == "true" ]]; then
    BASE_IMAGE=${BASE_IMAGE_CPU}
    LATEST_TAG=${LATEST_TAG_CPU}
else
    BASE_IMAGE=${BASE_IMAGE_GPU}
    LATEST_TAG=${LATEST_TAG_GPU}
fi

echo -e "${BLUE}Building Docker image '${REPO}:${LATEST_TAG}' from base image '${BASE_IMAGE}'...${NC}"

# --network host allows for the container's network stack to use the Docker host's network
docker build ${CONTEXT} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --tag "${REPO}:${LATEST_TAG}" \
    --network host \

if [[ ${PUSH_TO_GHCR} == "true" ]]; then
    echo -e "${BLUE}Pushing Docker image '${REPO}:${LATEST_TAG}' to GitHub Container Registry...${NC}"
    docker push ${REPO}:${LATEST_TAG}
fi
