ARG RUN_DIR="/var/task"
ARG RUN_USER="appuser"
ARG CONDA_VERSION="39_4.12.0"
ARG ARCH="x86_64"
#ARG CONDA_VERSION="39_23.1.0-1"

FROM python:3.9.15-bullseye as base
ARG RUN_DIR
ARG RUN_USER
ARG CONDA_VERSION
ARG ARCH

ENV APPUSER=${RUN_USER}
ENV MINICONDA_SCRIPT="Miniconda3-py${CONDA_VERSION}-Linux-${ARCH}.sh"
#https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-aarch64.sh
RUN apt-get update -y && \
    apt-get upgrade -y && \ 
    apt-get install -y python3-pip wget 
RUN mkdir -p ${RUN_DIR}
RUN groupadd --system --gid=9999  ${APPUSER} && \
    useradd --system --home-dir $RUN_DIR --uid=9999 --gid=${APPUSER} ${APPUSER}

RUN wget -O ${MINICONDA_SCRIPT} "https://repo.anaconda.com/miniconda/${MINICONDA_SCRIPT}" 

ENV CONDA_DIR=/var/conda
RUN sh ${MINICONDA_SCRIPT} -b -p $CONDA_DIR/miniconda -f 
ENV PATH=$CONDA_DIR/miniconda/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/miniconda/lib:$LD_LIBRARY_PATH
RUN eval "$(conda shell.bash hook)" && \
    chmod 755 $CONDA_DIR && \
    chown -R ${APPUSER}:${APPUSER} $CONDA_DIR && \
    chown -R ${APPUSER}:${APPUSER} $RUN_DIR && \
    conda init
RUN apt-get remove wget -y
RUN conda config --add channels conda-forge
RUN conda install mamba -c conda-forge -y

