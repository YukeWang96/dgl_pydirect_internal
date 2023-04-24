# conda activate dgl
rm logs/*
mkdir logs/
python 1_dgl_gcn.py > 1_dgl_gcn.log 2> 1_dgl_gcn.err
python 1_log2csv.py 1_dgl_gcn.log
# mv *.out logs
mv *.err logs/
mv *.log logs/