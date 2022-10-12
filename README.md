Profile pytorch-direct on snap

# profile
If you want to profile, the instructions are needed:

setup docker
```
docker pull happy233/glcc_pytorch_direct:new
docker run -it -v $(pwd):/pytorch-direct --gpus=all --shm-size="32g" happy233/glcc_pytorch_direct:new /bin/bash
docker run -it -v $(pwd):/pytorch-direct -v /data/datasets/graphs/osdi23_mgg:/pytorch-direct/graphdata --gpus=all happy233/glcc_pytorch_direct:new /bin/bash

```

<!-- docker run -it --gpus=all --shm-size="32g" happy233/glcc_pytorch_direct:base /bin/bash -->
<!-- docker pull zqj2333/pytorch-direct:v4 -->
<!-- docker exec -it profile /bin/bash -->
<!-- docker run  -it --gpus=all --shm-size="32g" -itd zqj2333/pytorch-direct:v4  /bin/bash -->

get graph dataset
```
cd dgl/examples/pytorch/graphsage/graphdata
wget https://storage.googleapis.com/graph_dataset/snap.tar.gz
tar -zxvf snap.tar.gz
```

setup profile
```
conda activate dgl
cd ../
python pytorchdirecttest.py
```

generate xls
```
cp *.out outfilefull/
cd xls_generate
python outtoxls.py
```

