import os

def main():

    gpu_list = [8]
    embedding_size=[16]

    mtx_path_list=[
        '../graphdata/reddit.mtx',
        '../graphdata/enwiki-2013.mtx',
        '../graphdata/it-2004.mtx',
        '../graphdata/papers100m.mtx',
        '../graphdata/ogbn-products.mtx',
        '../graphdata/ogbn-proteins.mtx',
        '../graphdata/com-orkut.mtx'
    ]

    dataset = [
            ( 'Reddit'                      , 602      	, 41),
            ( 'enwiki-2013'	                , 100	    , 12),   
            ( 'it-2004'                     , 128       , 172),
            ( 'paper100M'                   , 128       , 172),
            ( 'ogbn-products'	            , 100	    , 47),   
            ( 'ogbn-proteins'	            , 128		, 112),
            ( 'com-Orkut'		            , 128		, 128),
    ]


    for n in range(len(gpu_list)):
        ngpu=gpu_list[n]
        gpuinst='0'
        for i in range(1,ngpu):
            gpuinst=gpuinst+','+str(i)

        for i in range(len(mtx_path_list)):
            for k in range(len(embedding_size)):

                embed=embedding_size[k]
                source=mtx_path_list[i]
                for j in range(ngpu):
                    os.system('touch intermediate'+str(j)+'.out')
                
                # profile = "/opt/nvidia/nsight-compute/2022.3.0/ncu --metrics all --devices 0 "
                command='python gcn_multi_gpu.py --gpu '+gpuinst\
                        +' --graph-device cpu --data-device uva --source '+source\
                        +' --num-hidden '+str(embed) \
                        +' --nfeats '+str(dataset[i][1]) \
                        +' --out-classes '+str(dataset[i][2])
                # command = profile + command
                print(command)
                os.system(command)

                writefile=open(str(i+1)+'.out','a')
                writefile.write(source)
                writefile.write('\n')

                for j in range(ngpu):
                    myfile=open('intermediate'+str(j)+'.out','r')
                    writefile.write(str(ngpu)+' '+str(j)+' '+str(embed)+' ')
                    myline=myfile.readline()
                    myline=myfile.readline()
                    print(myline)
                    writefile.write(myline)
                    writefile.write('\n')
                    myfile.close()
                writefile.close()
                
                for j in range(ngpu):
                    os.system('rm intermediate'+str(j)+'.out')
                
if __name__ == '__main__':
    main()
