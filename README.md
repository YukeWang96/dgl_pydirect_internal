Profile pytorch-direct on snap

# profile
If you want to profile, the instructions are needed:

setup docker
```
docker push zqj2333/pytorch-direct:v4
docker run --name profile --gpus=all --shm-size="32g" -itd zqj2333/pytorch-direct:v4
docker exec -it profile /bin/bash
```

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
cp *.out outfilefull/
cd xls_generate
python outtoxls.py
```

