docker run --gpus device=0 \
           -it \
           -d \
	   --shm-size=6g \
           --name dynamic_iqa_ex \
           --mount type=bind,source=${HOME}/Documents/Datasets/iqa,dst=/dynamic_iqa_ex/Datasets \
           --mount type=bind,source=./,dst=/dynamic_iqa_ex \
           dynamic_iqa_ex:latest
