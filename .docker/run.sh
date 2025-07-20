docker run --gpus device=0 \
           -it \
           -d \
           --name dynamic_iqa_ex \
           --mount type=bind,source=${HOME}/Dev/Datasets,dst=/dynamic_iqa_ex/Datasets \
           --mount type=bind,source=./,dst=/dynamic_iqa_ex \
           --shm-size=6g \
           dynamic_iqa_ex:latest