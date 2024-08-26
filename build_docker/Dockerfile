# Use the official Ubuntu 22.04 image as a base
FROM ubuntu:22.04

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libffi-dev \
    libssl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install the specified Python packages
RUN pip install torch==1.12.0 torchvision==0.13.0 numpy==1.23 torchtext==0.4.0 spacy==3.4.1 scipy==1.9.0 transformers==4.25.1 opencv-python==4.8.0.76
RUN pip install ffmpeg-python==0.2.0 face_alignment==1.3.5
RUN pip install scikit-learn==1.2.0 matplotlib==3.5.2

# Copy the test.py file into the container
COPY test_bkp.py /app/test_bkp.py
COPY fid_score.py /app/fid_score.py
COPY npy2vid.py /app/npy2vid.py
COPY src/ /app/src/
COPY thirdparty /app/thirdparty
COPY navigator_normal_new.pth /app/
COPY embsmall_bet/ /app/embsmall_bet
COPY test_repro/ /app/test_repro
COPY utils/ /app/utils
COPY models/ /app/models
COPY data/ /app/data
COPY checkpoints_ae2 /app/checkpoints_ae2
COPY examples/ /app/examples
COPY face_image.jpg /app/face_image.jpg
# Set the working directory
WORKDIR /app

# Run the test.py script
CMD ["sh", "-c", "python3 test_bkp.py && python3 fid_score.py && python3 npy2vid.py"]
