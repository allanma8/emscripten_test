FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

#ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

## Optionally install the cmake for vcpkg
#COPY ./reinstall-cmake.sh /tmp/
#
#RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
#        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
#    fi \
#    && rm -f /tmp/reinstall-cmake.sh
#
# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install your-port-name-here"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends your-package-list-here

ENV DEBIAN_FRONTEND=noninteractive

# This should be the same as what ever we are using for onnxruntime
ARG EMSDK_VERSION=3.1.57

RUN apt-get update && apt-get install -y \
    curl \
    git \
    cmake \
    clang \
    build-essential \
    ca-certificates \
    python3 \
    python3-dev \
    python3-pip \
    && \
    rm -rf /var/lib/apt/lists/*

ENV NODE_VERSION=22
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash

ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"

RUN git clone https://github.com/emscripten-core/emsdk.git && \
    cd emsdk && \
    ./emsdk install ${EMSDK_VERSION} && \
    ./emsdk activate ${EMSDK_VERSION}

#ARG USER=dev
#RUN useradd --groups sudo --no-create-home --shell /bin/zsh ${USER} \
#    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
#    && chmod 0440 /etc/sudoers.d/${USER}
#
#USER ${USER}
#WORKDIR /home/${USER}
#
#RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
#ENV NVM_DIR=/home/${USER}/.nvm
#
#RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
#RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
#RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
#ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"