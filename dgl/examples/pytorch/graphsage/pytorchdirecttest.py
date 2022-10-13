import os
import xlwt

def main():

    mtx_path_list=[

    # 'graphdata/reddit.mtx',
    # 'graphdata/enwiki-2013.mtx',
    # 'graphdata/it-2004.mtx',
    'graphdata/papers100m.mtx',
    # 'graphdata/ogbn-products.mtx',
    # 'graphdata/ogbn-proteins.mtx',
    # 'graphdata/com-orkut.mtx',

    # 'graphdata/snap/email-Eu-core/email-Eu-core.mtx',
    # 'graphdata/snap/email-Eu-core/email-Eu-core.mtx',
    # 'graphdata/snap/wiki-talk-temporal/wiki-talk-temporal.mtx',
    # 'graphdata/snap/soc-LiveJournal1/soc-LiveJournal1.mtx',
    # 'graphdata/snap/com-LiveJournal/com-LiveJournal.mtx',
    # 'graphdata/snap/cit-Patents/cit-Patents.mtx',
    # 'graphdata/snap/sx-stackoverflow/sx-stackoverflow.mtx',
    # 'graphdata/snap/wiki-Talk/wiki-Talk.mtx',
    # 'graphdata/snap/roadNet-CA/roadNet-CA.mtx',
    # 'graphdata/snap/wiki-topcats/wiki-topcats.mtx',
    # 'graphdata/snap/as-Skitter/as-Skitter.mtx',
    # 'graphdata/snap/roadNet-TX/roadNet-TX.mtx',
    # 'graphdata/snap/papers100m.mm',
    # 'graphdata/snap/mag240m.mm',
    # 'graphdata/snap/twitter7.mm',
    # 'graphdata/snap/uk-2006-05.mm'
    ]
    #ngpu=1
    gpuinst='0'
    # gpu_list=[2,4,8]
    # embedding_size=[16,32,64,128,256]
    gpu_list=[4]
    # gpu_list = [1]
    embedding_size=[16]
    
    for n in range(1):
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
                
                command='python train_sampling_multi_gpu.py --gpu '+gpuinst+' --graph-device cpu --data-device uva --source '+source+' --nfeats '+str(embed)
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
                    print('content')
                    print(myline)
                    writefile.write(myline)
                    writefile.write('\n')
                    myfile.close()
                writefile.close()
                
                for j in range(ngpu):
                    os.system('rm intermediate'+str(j)+'.out')

if __name__ == '__main__':
    main()
