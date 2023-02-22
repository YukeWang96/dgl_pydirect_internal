docker run -it -v $(pwd):/pytorch-direct \
            -v /home/yukewang/glcc-pydirect/dgl/examples/pytorch/graphsage/graphdata:/pytorch-direct/graphdata \
            -w /pytorch-direct \
            --gpus=all \
            --shm-size="128g" happy233/glcc_pytorch_direct:new /bin/bash

# 64g
# 16g