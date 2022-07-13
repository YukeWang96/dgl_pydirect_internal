Profile pytorch-direct on snap

# profile
If you want to profile, the instructions are needed:
```
docker push zqj2333/pytorch-direct:v4
docker run --name profile --gpus=all --shm-size="32g" -itd zqj2333/pytorch-direct:v4
docker exec -it profile /bin/bash


conda activate dgl
cd dgl/examples/pytorch/graphsage/graphdata
source download_SNAP.sh
cd ../
python pytorchdirecttest.py
cp *.out outfilefull/
cd xls_generate
python outtoxls.py
```

