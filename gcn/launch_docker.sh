docker run -it -v $(pwd):/pytorch-direct \
            -v $pwd:/pytorch-direct/graphdata \
            -w /pytorch-direct \
            --gpus=all \
            --shm-size="512g" happy233/glcc_pytorch_direct:new /bin/bash

# 64g
# 16g