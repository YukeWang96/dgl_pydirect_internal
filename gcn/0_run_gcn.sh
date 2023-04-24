# conda activate dgl
rm logs/*
mkdir logs/
/root/anaconda3/envs/dgl/bin/python 1_dgl_gcn.py > 1_dgl_gcn.log 2> 1_dgl_gcn.err
/root/anaconda3/envs/dgl/bin/python 1_log2csv.py 1_dgl_gcn.log
# mv *.out logs
mv *.err logs/
mv *.log logs/