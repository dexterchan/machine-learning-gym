ARG RUN_DIR="/var/task"
ARG RUN_USER="appuser"
ARG CONDA_VERSION="39_23.3.1-0"
ARG CONTAINER_VERSION="3.9-slim-bullseye"


FROM python:${CONTAINER_VERSION} as base
ARG RUN_DIR
ARG RUN_USER
ARG CONDA_VERSION
ARG ARCH

ENV APPUSER=${RUN_USER}

RUN apt-get update -y && \
    apt-get upgrade -y 

RUN mkdir -p ${RUN_DIR}
RUN groupadd --system --gid=9999  ${APPUSER} && \
    useradd --system --home-dir $RUN_DIR --uid=9999 --gid=${APPUSER} ${APPUSER}

# Debian python install pip
RUN apt-get install -y python3-pip 

FROM base as conda_install
ARG RUN_DIR
ARG RUN_USER
ARG CONDA_VERSION
ARG ARCH
ENV MINICONDA_SCRIPT="Miniconda3-py${CONDA_VERSION}-Linux-${ARCH}.sh"

# Install miniconda
#https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-aarch64.sh
#https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
ENV CONDA_DIR=/var/conda
RUN mkdir -p $CONDA_DIR && \
    apt-get install -y wget && \
    wget -O ${MINICONDA_SCRIPT} "https://repo.anaconda.com/miniconda/${MINICONDA_SCRIPT}" && \
    apt-get remove -y wget && \
    sh ${MINICONDA_SCRIPT} -b -p $CONDA_DIR/miniconda -f && \
    rm -Rf ${MINICONDA_SCRIPT}

ENV PATH=$CONDA_DIR/miniconda/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/miniconda/lib:$LD_LIBRARY_PATH
RUN chown -R ${APPUSER}:${APPUSER} $CONDA_DIR
# RUN eval "$(conda shell.bash hook)" && \
#     chmod 755 $CONDA_DIR && \
#     chown -R ${APPUSER}:${APPUSER} $CONDA_DIR && \
#     conda init
RUN chown -R ${APPUSER}:${APPUSER} $RUN_DIR

USER ${APPUSER}
RUN conda config --add channels conda-forge
RUN conda install mamba -c conda-forge -y

FROM conda_install as install
ARG RUN_DIR
ARG RUN_USER
ARG ARCH
ARG PRIVATE_CONDA_CHANNEL

WORKDIR $RUN_DIR
# COPY requirements.txt and modify to RUN_USER as owner
COPY --chown=${RUN_USER}:${RUN_USER} requirements-linux-${ARCH}.txt ${RUN_DIR}/requirements-linux-${ARCH}.txt

# Install requirements
RUN conda config --add channels ${PRIVATE_CONDA_CHANNEL}
#reference https://uwekorn.com/2021/03/01/deploying-conda-environments-in-docker-how-to-do-it-right.html
RUN mamba create --copy -p $CONDA_DIR/env --file requirements-linux-${ARCH}.txt && conda clean -afy

FROM base as s3fs
RUN apt -y install s3fs

FROM s3fs as app
ARG RUN_DIR
ARG RUN_USER
ARG ARCH
ENV CONDA_DIR=/var/conda
ENV PATH=$CONDA_DIR/env/bin:$PATH
ENV LD_LIBRARY_PATH=$CONDA_DIR/env/lib:$LD_LIBRARY_PATH

COPY --chown=${RUN_USER}:${RUN_USER} src/q_learning_lab ${RUN_DIR}/q_learning_lab
COPY --chown=${RUN_USER}:${RUN_USER} scripts/config/*.* ${RUN_DIR}/config/samples/
COPY --chown=${RUN_USER}:${RUN_USER} scripts/container/mount_s3fs.sh ${RUN_DIR}/mount_s3fs.sh
COPY --chown=${RUN_USER}:${RUN_USER} scripts/execute_s3_training.sh ${RUN_DIR}/execute_s3_training.sh
ENV PYTHONPATH="${PYTHONPATH}:${RUN_DIR}"
RUN mkdir -p ${RUN_DIR}/workspaces
WORKDIR ${RUN_DIR}/workspaces

FROM app as dev
ARG RUN_USER
COPY --chown=${RUN_USER}:${RUN_USER} --from=install ${CONDA_DIR}/env ${CONDA_DIR}/env


FROM install as install_jupyter
ARG RUN_DIR
ARG RUN_USER
ARG ARCH
ARG PRIVATE_CONDA_CHANNEL

COPY --chown=${RUN_USER}:${RUN_USER} requirements_jupyter.txt ${RUN_DIR}/requirements_jupyter.txt
RUN mamba create --copy -p $CONDA_DIR/env --file requirements_jupyter.txt && conda clean -afy

FROM app as jupyter
ARG RUN_USER
COPY --chown=${RUN_USER}:${RUN_USER} --from=install_jupyter ${CONDA_DIR}/env ${CONDA_DIR}/env
