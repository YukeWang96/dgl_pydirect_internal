cd snap/

wget https://sparse.tamu.edu/MM/SNAP/email-Eu-core.tar.gz

find . -name '*.tar.gz' -exec tar xvf {} \;
rm *.tar.gz
cp ../conv.c .
gcc -O3 -o conv conv.c

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
mv ${ii}.mtx ${ii}.mt0
../conv ${ii}.mt0 ${ii}.mtx
rm ${ii}.mt0
cd ..
done

rm conv conv.c
