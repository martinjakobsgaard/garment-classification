#!/bin/bash

docker kill tfserv
docker container rm tfserv

docker run -p 8501:8501 --name tfserv \
--mount type=bind,source=/home/inwatec/workspace/garment-classification/models/resnet50-server,target=/models/resnet-serving \
-e MODEL_NAME=resnet-serving -t tensorflow/serving
