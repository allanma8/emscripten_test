FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive

# Don't bump this until https://github.com/llvm/llvm-project/issues/107685 has been fixed.
# Clang 20.0 doesn't work well with emscripten at the moment and we get linker crash issues.
ARG EMSDK_VERSION=3.1.59

RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    cmake \
    clang \
    build-essential \
    ca-certificates \
    python3 \
    python3-dev \
    python3-pip \
    openjdk-11-jre-headless \
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