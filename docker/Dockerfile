FROM ubuntu:22.04 AS stage1

ARG AMDGPU_INSTALLER=https://artifactory-cdn.amd.com/artifactory/list/amdgpu-deb/amdgpu-install-internal_6.5-22.04-1_all.deb
ARG AMD_BUILD_ID=2152768
ARG ROCM_BUILD_ID=compute-rocm-dkms-no-npi-hipclang/15896

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    sed -i -e 's/archive\.ubuntu\.com/cn\.archive\.ubuntu\.com/' /etc/apt/sources.list && \
    apt update && apt install -y rsync sqlite3 wget && \
    wget -O /tmp/amdgpu-install.deb ${AMDGPU_INSTALLER} && \
    apt install -y /tmp/amdgpu-install.deb && \
    rm -f /tmp/amdgpu-install.deb && \
    sed -i 's/compute-cdn\.amd\.com/localhost/g' /usr/bin/amdgpu-repo && \
    amdgpu-repo --amdgpu-build=${AMD_BUILD_ID} --rocm-build=${ROCM_BUILD_ID}


FROM stage1 AS stage1_1

ARG GFX_ARCH=gfx942

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt update && \
    amdgpu-install --usecase=rocm --no-dkms -y && \
    find /opt/rocm/lib -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/hipblaslt/library -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/rocblas/library -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/share/miopen/db -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    sqlite3 /opt/rocm/lib/rocfft/rocfft_kernel_cache.db "delete from cache_v1 where arch != '${GFX_ARCH}' ; vacuum"

ENV ROCM_HOME=/opt/rocm
ENV CPLUS_INCLUDE_PATH=/opt/rocm/include:
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH


FROM stage1 AS download_whls

ARG EXTRA_DIR=
ARG WHL_URL_PREFIX=https://compute-artifactory.amd.com/artifactory/compute-pytorch-rocm/compute-rocm-dkms-no-npi-hipclang/15896
ARG APEX_WHL=apex-1.6.0+rocm6.5.0.git69ee42b2-cp310-cp310-linux_x86_64.whl
ARG TORCH_WHL=torch-2.6.0+rocm6.5.0.git9d0a4a1a-cp310-cp310-linux_x86_64.whl
ARG TORCHVISION_WHL=torchvision-0.21.0+rocm6.5.0.git4040d51f-cp310-cp310-linux_x86_64.whl
ARG TRITON_ROCM_WHL=pytorch_triton_rocm-3.2.0+rocm6.5.0.git6da9e660-cp310-cp310-linux_x86_64.whl
ARG CMAKE_VERSION=3.26.4

WORKDIR /install

RUN wget -nv -O /install/${APEX_WHL} ${WHL_URL_PREFIX}/${EXTRA_DIR}${APEX_WHL}
RUN wget -nv -O /install/${TORCH_WHL} ${WHL_URL_PREFIX}/${EXTRA_DIR}${TORCH_WHL}
RUN wget -nv -O /install/${TORCHVISION_WHL} ${WHL_URL_PREFIX}/${EXTRA_DIR}${TORCHVISION_WHL}
RUN wget -nv -O /install/${TRITON_ROCM_WHL} ${WHL_URL_PREFIX}/${EXTRA_DIR}${TRITON_ROCM_WHL}

# wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz
COPY cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz /install/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz


FROM stage1_1 AS stage2

ARG APEX_WHL=apex-1.6.0+rocm6.5.0.git69ee42b2-cp310-cp310-linux_x86_64.whl
ARG TORCH_WHL=torch-2.6.0+rocm6.5.0.git9d0a4a1a-cp310-cp310-linux_x86_64.whl
ARG TORCHVISION_WHL=torchvision-0.21.0+rocm6.5.0.git4040d51f-cp310-cp310-linux_x86_64.whl
ARG TRITON_ROCM_WHL=pytorch_triton_rocm-3.2.0+rocm6.5.0.git6da9e660-cp310-cp310-linux_x86_64.whl
ARG CMAKE_VERSION=3.26.4
ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt update && \
    apt install -y git software-properties-common curl rsync dialog gfortran

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    if ! python3 --version | grep -q ${PYTHON_VERSION} ; then \
    add-apt-repository -y ppa:deadsnakes/ppa && apt update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
                       python${PYTHON_VERSION}-lib2to3 python-is-python3 ; fi

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

# workaround: if we use setuptools v80 we'll get 'torch' not found when building aiter
RUN --mount=type=bind,from=download_whls,source=/install,target=/tmp/install \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip "setuptools<80" wheel numpy einops ninja && \
        pip install /opt/rocm/share/amd_smi && \
    pip install /tmp/install/${TORCH_WHL} /tmp/install/${TRITON_ROCM_WHL}        \
                /tmp/install/${TORCHVISION_WHL}   \
                /tmp/install/${APEX_WHL} && \
    tar zfxv /tmp/install/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -C /opt/ && \
        mv /opt/cmake-${CMAKE_VERSION}-linux-x86_64 /opt/cmake-${CMAKE_VERSION}

ENV PATH=/opt/cmake-${CMAKE_VERSION}/bin:$PATH
ENV PYTORCH_ROCM_ARCH=gfx942

ENV TORCH_BLAS_PREFER_HIPBLASLT=1
ENV ROCBLAS_USE_HIPBLASLT=1
ENV DISABLE_ADDMM_HIP_LT=0
ENV HIPBLASLT_ALLOW_TF32=1
ENV PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM=1
ENV PYTORCH_MIOPEN_SUGGEST_NHWC=1
