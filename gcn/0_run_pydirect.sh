# conda activate dgl
python 1_pytorchdirecttest.py > 1_pytorchdirecttest.log 2> 1_pytorchdirecttest.err
python 1_log2csv.py 1_pytorchdirecttest.log
mv *.out logs
mv *.err logs
mv *.log logs