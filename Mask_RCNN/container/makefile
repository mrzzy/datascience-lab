#
# makefile
# Container Makefile
# 

DOCKER_USERNAME:=mrzzy
CONTAINER_TAG:=mrcnn-test
PORT_MAPS:=6006:6006 8008:8008

# Targets
.DEFAULT: all
.PHONY: all build run push

all: build run

# build mrcnn container 
build:
	docker build -t $(CONTAINER_TAG) -f container/Dockerfile .

# run mrcnn container as current user
run:
	-docker run --runtime=nvidia \
		-v $(shell pwd):/mask_rcnn_test \
		$(patsubst %, -p %, $(PORT_MAPS)) \
		-it $(CONTAINER_TAG)

# push mrcnn container to dockerhub
push:
	docker login
	docker tag $(CONTAINER_TAG) $(DOCKER_USERNAME)/$(CONTAINER_TAG)
	docker push $(DOCKER_USERNAME)/$(CONTAINER_TAG)
