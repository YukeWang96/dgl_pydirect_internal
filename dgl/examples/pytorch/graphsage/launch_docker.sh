docker run -it -v $(pwd):/pytorch-direct \
            -v /data/datasets/graphs/osdi23_mgg:/pytorch-direct/graphdata \
            -w /pytorch-direct --gpus=all --shm-size="16g" happy233/glcc_pytorch_direct:new /bin/bash