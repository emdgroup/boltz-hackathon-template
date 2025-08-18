# Based on Liana64's contribution https://github.com/jwohlwend/boltz/blob/5ee0e6b9740b85ff24349aacc4d69615f499490b/Dockerfile
ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=23.3.1-0
ARG BASE_IMAGE=nvidia/cuda:12.3.0-runtime-ubuntu22.04

FROM ${BASE_IMAGE} AS builder
ARG MINIFORGE_NAME
ARG MINIFORGE_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  build-essential \
  python3 \
  python3-venv \
  python3-dev \
  wget \
  && wget --no-check-certificate --no-hsts https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O miniforge.sh \
  && bash miniforge.sh -b -p /opt/conda \
  && rm miniforge.sh \
  && /opt/conda/bin/mamba init bash

WORKDIR /app
COPY environment.yml /app/
COPY src /app/src
COPY pyproject.toml /app/pyproject.toml

RUN /opt/conda/bin/mamba env create -f environment.yml --name boltz && \
    /opt/conda/bin/mamba init bash && \
    /opt/conda/bin/conda run -n boltz pip install --no-cache-dir --upgrade pip && \
    /opt/conda/bin/conda run -n boltz pip install --no-cache-dir .[cuda] && \
    apt-get purge -y git build-essential wget && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

FROM ${BASE_IMAGE}
COPY --from=builder /opt/conda /opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  build-essential \
  python3-dev \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH="/opt/conda/bin:$PATH" \
  LANG=C.UTF-8 \
  PYTHONUNBUFFERED=1

ARG USERNAME=boltz
ARG UID=900
ARG GID=900
RUN groupadd --gid $GID $USERNAME && \
  useradd --uid $UID --gid $GID --create-home --shell /bin/bash $USERNAME

WORKDIR /app

# Copy everything
COPY . /app/

RUN chown -R $USERNAME:$USERNAME /app

USER $USERNAME

# Initialize mamba and activate the boltz environment
SHELL ["/bin/bash", "-c"]
RUN mamba init bash && \
    echo "mamba activate boltz" >> ~/.bashrc

CMD ["/bin/bash"]