FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

ARG EMSCRIPTEN_VERSION=3.1.59
ENV EMSDK /emsdk

# Install dependencies needed to install/build emsdk
RUN apt-get -qq -y update \
    && apt-get -qq install -y --no-install-recommends \
        curl \
        file \
        git \
        binutils \
        build-essential \
        ca-certificates \
        python3 \
        python3-pip

# Install emsdk
RUN git clone https://github.com/emscripten-core/emsdk.git ${EMSDK} && \
    cd ${EMSDK} && \
    ./emsdk install ${EMSCRIPTEN_VERSION} && \
    ./emsdk activate ${EMSCRIPTEN_VERSION}

# Clean up some files from emsdk
RUN cd ${EMSDK} && . ./emsdk_env.sh \
    && strip -s `which node` \
    && rm -fr ${EMSDK}/upstream/emscripten/tests \
    && find ${EMSDK}/upstream/bin -type f -exec strip -s {} + || true

#
# This is the actual image we will be using to run stuff in
#
FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04 as runner

COPY --from=builder /emsdk /emsdk

# ARG USERNAME=dev
#
# ARG USER_UID=1001
# ARG USER_GID=$USER_UID

ENV EMSDK=/emsdk \
    EMSDK_NODE="/emsdk/node/18.20.3_64bit/bin/node" \
    PATH="/emsdk:/emsdk/upstream/emscripten:/emsdk/node/18.20.3_64bit/bin:${PATH}"

RUN apt-get -qq -y update \
    && DEBIAN_FRONTEND="noninteractive" TZ="America/San_Francisco" apt-get -qq install -y --no-install-recommends \
        sudo \
        libxml2 \
        ca-certificates \
        python3 \
        python3-pip \
        wget \
        curl \
        zip \
        unzip \
        git \
        git-lfs \
        ssh-client \
        build-essential \
        ninja-build \
        ant \
        libidn12 \
        cmake \
        openjdk-11-jre-headless \
    && apt-get -y clean \
    && apt-get -y autoclean \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/debconf/*-old \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/?? \
    && rm -rf /usr/share/man/??_*

# Create user
#RUN groupadd --gid $USER_GID $USERNAME \
#    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#    && chmod 0440 /etc/sudoers.d/$USERNAME
#
#USER $USERNAME

# HACK FIX
# This removes the dependency on `google-closure-compiler-linux` which causes issues for arm64 things
RUN cd ${EMSDK}/upstream/emscripten \
    && npm uninstall google-closure-compiler-linux \
    && npm install