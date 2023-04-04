# conda activate dgl
python 1_dgl_gin.py > 1_dgl_gin.log 2> 1_dgl_gin.err
python 1_log2csv.py 1_dgl_gin.log
mv *.out logs
mv *.err logs
mv *.log logs