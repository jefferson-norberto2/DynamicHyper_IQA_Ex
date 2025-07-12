FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Atualiza e instala ferramentas básicas
RUN apt update && \
    apt install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /dynamic_iqa_ex/requirements.txt

# Instala dependências Python
RUN pip install --upgrade pip && \
    pip install -r /dynamic_iqa_ex/requirements.txt \
    pip cache purge

# Baixa e compila o Apex
RUN mkdir -p /ddfnet && \
    cd /ddfnet && \
    wget https://github.com/NVIDIA/apex/archive/refs/tags/25.04.zip && \
    unzip 25.04.zip && \
    mv apex-25.04 apex && \
    cd apex && \
    python setup.py install --cpp_ext --cuda_ext && \
    cd ../ && rm -rf 25.04.zip apex
    
WORKDIR /dynamic_iqa_ex

CMD ["bash"]