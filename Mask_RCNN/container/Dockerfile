#
# Dockerfile
# Builds a GPU enabled container to train and run maskrcnn models 
# with Python3 and Jupyter notebook support
#

FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

# Expose ports
# tensorboard
EXPOSE 6006
# jupyter notebook
EXPOSE 8008

# Setup project directory 
RUN mkdir /mask_rcnn_test
WORKDIR /mask_rcnn_test

# Install Project dependencies
RUN apt-get update &&  apt-get install -y \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /mask_rcnn_test/
RUN pip3 --no-cache-dir install -r requirements.txt

# Install Project Dependency: pycocotools
RUN cd /tmp && git clone https://github.com/pdollar/coco.git \
    && cd coco/PythonAPI \
    && make \
    && make install \
    && python3 setup.py install 

# Retrieve maskrcnn mscoco pretrained model
#RUN curl -L "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5" \
#    -O ./mask_rcnn_coco.h5
COPY mask_rcnn_coco.h5 /mask_rcnn_test/

# Copy project files
COPY datasets /mask_rcnn_test/datasets
COPY mrcnn /mask_rcnn_test/mrcnn
COPY samples /mask_rcnn_test/samples
COPY setup.py eval.py /mask_rcnn_test/

CMD bash
