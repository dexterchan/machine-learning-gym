ARG RUN_DIR="/var/task"
ARG RUN_USER="appuser"
ARG CONDA_VERSION="39_4.12.0"
ARG CONTAINER_VERSION="3.9.15-bullseye"


FROM python:${CONTAINER_VERSION} as base
ARG RUN_DIR
ARG RUN_USER
ARG CONDA_VERSION
ARG ARCH

ENV APPUSER=${RUN_USER}
ENV MINICONDA_SCRIPT="Miniconda3-py${CONDA_VERSION}-Linux-${ARCH}.sh"
#https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-aarch64.sh
# Alpine update
RUN apt-get update -y && \
    apt-get upgrade -y 

RUN mkdir -p ${RUN_DIR}
RUN groupadd --system --gid=9999  ${APPUSER} && \
    useradd --system --home-dir $RUN_DIR --uid=9999 --gid=${APPUSER} ${APPUSER}

# Debian python install pip
RUN apt-get install -y python3-pip 

# Install miniconda
ENV CONDA_DIR=/var/conda
RUN mkdir -p $CONDA_DIR && \
    apt-get install -y wget && \
    wget -O ${MINICONDA_SCRIPT} "https://repo.anaconda.com/miniconda/${MINICONDA_SCRIPT}" && \
    apt-get remove -y wget && \
    sh ${MINICONDA_SCRIPT} -b -p $CONDA_DIR/miniconda -f && \
    rm -Rf ${MINICONDA_SCRIPT}

ENV PATH=$CONDA_DIR/miniconda/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/miniconda/lib:$LD_LIBRARY_PATH
RUN eval "$(conda shell.bash hook)" && \
    chmod 755 $CONDA_DIR && \
    chown -R ${APPUSER}:${APPUSER} $CONDA_DIR && \
    conda init
RUN chown -R ${APPUSER}:${APPUSER} $RUN_DIR

USER ${APPUSER}
RUN conda config --add channels conda-forge
RUN conda install mamba -c conda-forge -y

FROM base as install
ARG RUN_DIR
ARG RUN_USER
ARG ARCH

WORKDIR $RUN_DIR
# COPY requirements.txt and modify to RUN_USER as owner
COPY --chown=${RUN_USER}:${RUN_USER} requirements-linux-${ARCH}.txt ${RUN_DIR}/requirements-linux-${ARCH}.txt

# Install requirements
RUN mamba install -y -c conda-forge --file requirements-linux-${ARCH}.txt
RUN pip install -y pygame==2.3.0
# RUN	mamba install -y -c conda-forge --file requirements_jupyter.txt
# RUN pip install -r requirements-linux-${ARCH}.txt

FROM install as dev
ARG RUN_DIR
ARG RUN_USER
ARG ARCH

#COPY src/q_learning_lab module into workdir
COPY --chown=${RUN_USER}:${RUN_USER} src/q_learning_lab ${RUN_DIR}/q_learning_lab
#COPY sample config into workdir
COPY --chown=${RUN_USER}:${RUN_USER} scripts/config/*.* ${RUN_DIR}/config/samples/
ENV PYTHONPATH="${PYTHONPATH}:${RUN_DIR}"
RUN mkdir -p ${RUN_DIR}/workspaces
WORKDIR ${RUN_DIR}/workspaces