import os
import xlwt

def findAllfile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield base+'/'+f

def main():
    warp_table=[1,2,4,8,16,32]
    base = '../warpout'

    workbook=xlwt.Workbook()
    worksheet=workbook.add_sheet('aggrowwarp')
    itr=0
    for i in findAllfile(base):
        outfile=open(i,'r')
        for j in range(2):
            for k in range(3):
                for m in range(6):

                    log=outfile.readline()
                    path=log.split('/')
                    graph_name=path[-1]
                    log=outfile.readline()
                    log=log.split()
                    worksheet.write(itr+1,0,graph_name)
                    worksheet.write(itr+1,1,log[0])
                    worksheet.write(itr+1,2,log[1])
                    worksheet.write(itr+1,3,log[2])
                    worksheet.write(itr+1,4,log[3])
                    worksheet.write(itr+1,5,log[4])
                    worksheet.write(itr+1,6,log[5])
                    #worksheet.write(itr+1,6,log[6])
                    itr=itr+1
                    log=outfile.readline()
        outfile.close()
    workbook.save('aggrowwarp.xls')
    

if __name__ == '__main__':
    main()
