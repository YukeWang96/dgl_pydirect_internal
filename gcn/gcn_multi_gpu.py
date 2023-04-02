import os
import dgl
import torch as th
import torch.nn.functional as F
import dgl.multiprocessing as mp
import time
import argparse
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from model import SAGE, AGNN
import statistics
import warnings
warnings.filterwarnings("ignore")

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id):
    """
    Extracts features and labels for a subset of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(dev_id)
    batch_labels = labels[seeds].to(dev_id)
    return batch_inputs, batch_labels

#### Entry point
def run(proc_id, n_gpus, args, devices, data, my_batch_size):
    # Start up distributed training, if enabled.
    device = th.device(devices[proc_id])
    #set_target_gpu(device)
    if n_gpus > 0:
        th.cuda.set_device(device)
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)

    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, val_nfeat, test_nfeat, \
    train_labels, val_labels, test_labels, train_nid, val_nid, test_nid = data

    if args.data_device == 'gpu':
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)
    elif args.data_device == 'uva':
        cudart = th.cuda.cudart()
        cudart.cudaHostRegister(train_nfeat.data_ptr(),
            train_nfeat.numel() * train_nfeat.element_size(), 0)
        cudart.cudaHostRegister(train_labels.data_ptr(),
            train_labels.numel() * train_labels.element_size(), 0)
        #train_nfeat = dgl.contrib.UnifiedTensor(train_nfeat, device=device)
        #train_labels = dgl.contrib.UnifiedTensor(train_labels, device=device)

    in_feats = train_nfeat.shape[1]

    if args.graph_device == 'gpu':
        train_nid = train_nid.to(device)
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        args.num_workers = 0
    elif args.graph_device == 'uva':
        train_nid = train_nid.to(device)
        train_g.pin_memory_()
        args.num_workers = 0

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    # sampler = dgl.dataloading.NeighborSampler([25])

    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        use_ddp=n_gpus > 1,
        #device=device,
        batch_size=my_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
        #use_uva=True
        )

    # Define model and optimizer
    # model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = AGNN(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)

    #model=nn.Linear(4,2,True)
    model = model.to(device)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    # loss_fcn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #if proc_id==0:
    #    print("memory")
    #    print(th.cuda.memory_allocated(proc_id))
    #    print(th.cuda.memory_reserved(proc_id))
    # Training loop
    avg = 0
    avg_fetch=0
    avg_agg=0
    iter_tput = []
    iter_time = []
    for epoch in tqdm(range(args.num_epochs)):
        # if proc_id==0:
            # print(str(epoch), end=" ")
        fetch = 0
        agg = 0
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        #reporter = MemReporter()
        #reporter.report()
        # if proc_id==0:
        #     print("ready to dataloader")
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # if proc_id==0:
            #     print("dataloader success")
            # print(blocks[0])
            #reporter = MemReporter()
            #reporter.report()
            blocks = [block.int().to(device) for block in blocks] # move graph structure.
            
            th.distributed.barrier()
            if proc_id == 0:
                th.cuda.synchronize()
                #print("memory")
                #print(th.cuda.memory_allocated(proc_id))
                #print(th.cuda.memory_reserved(proc_id))
                tic_step = time.time()
            #print(input_nodes.shape)
            # Load the input features as well as output labels
            #with th.no_grad():

            th.cuda.synchronize()

            start_fetch=time.time()
            #reporter = MemReporter()
            #reporter.report()
            with th.no_grad():
                batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
             
                th.cuda.synchronize()
                end_fetch = time.time()
                fetch=fetch + end_fetch - start_fetch
                # print("batch_inputs: ", batch_inputs)
                # print("blocks: ", blocks)
                #reporter = MemReporter()
                #reporter.report()
                #th.cuda.empty_cache()
                #print("memory1")
                #print(th.cuda.memory_allocated(proc_id))
                #print(th.cuda.memory_reserved(proc_id))
                start_agg = time.time()
                # Compute loss and prediction
                #with th.no_grad():
                # print("modelstart")
                batch_pred = model(blocks, batch_inputs)
                # print("modelend")
                th.cuda.synchronize()
                end_agg = time.time()
                agg = agg + end_agg - start_agg
                #print(agg_profile)
                #print(batch_inputs.shape)
                #print(batch_pred.shape)
            # th.cuda.synchronize()
            # end_agg=time.time()
            #agg=agg+end_agg-start_agg
            # agg=agg+agg_profile
            #loss = loss_fcn(batch_pred, batch_labels)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            
            th.distributed.barrier()
            if proc_id == 0:
                dur = time.time() - tic_step
                iter_tput.append(len(seeds) * n_gpus / (dur))
                iter_time.append(dur)
            #if step % args.log_every == 0 and proc_id == 0:
               # acc = compute_acc(batch_pred, batch_labels)
                #print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                 #   epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))
            break

        if epoch >= 1:
            avg_fetch = avg_fetch + fetch
            avg_agg = avg_agg + agg
        # print("inter end")
        #reporter = MemReporter()
        #reporter.report()
        if n_gpus > 1:
            th.distributed.barrier()
        toc = time.time()

    outfile=open('intermediate'+str(proc_id)+'.out','a')
    outfile.write(str("{:.3f}".format(avg_fetch / (args.num_epochs - 1) * 1e3)) + " ")
    outfile.write(str("{:.3f}".format(avg_agg / (args.num_epochs - 1) * 1e3)))
    outfile.close()

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        #print('Avg epoch time: {}'.format(avg / (epoch - 4)))
        fetch_avg = avg_fetch / (args.num_epochs - 1) * 1e3
        agg_avg = avg_agg / (args.num_epochs - 1) * 1e3
        print('Fetch (ms):\t{:.3f}'.format(fetch_avg))
        print('Aggre (ms):\t{:.3f}'.format(agg_avg))
        print('Total (ms):\t{:.3f}'.format(fetch_avg + agg_avg))
        print('Avg Iter (ms):\t{:.3f}'.format(sum(iter_time)/len(iter_time)*1e3))
        print('MIN Iter (ms):\t{:.3f}'.format(min(iter_time)*1e3))
        print('MED Iter (ms):\t{:.3f}'.format(statistics.median(iter_time)*1e3))
        print('MAX Iter (ms):\t{:.3f}'.format(max(iter_time)*1e3))
        print([i*1e3 for i in iter_time])

if __name__ == '__main__':
    #th.multiprocessing.set_start_method('spawn')
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=1)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--graph-device', choices=('cpu', 'gpu', 'uva'), default='cpu',
                           help="Device to perform the sampling. "
                                "Must have 0 workers for 'gpu' and 'uva'")
    argparser.add_argument('--data-device', choices=('cpu', 'gpu', 'uva'), default='gpu',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "Use 'cpu' to keep the features on host memory and "
                                "'uva' to enable UnifiedTensor (GPU zero-copy access on "
                                "pinned host memory).")
    argparser.add_argument('--source', type=str, default='error')
    argparser.add_argument('--nfeats', type=int, default=16)
    args = argparser.parse_args()

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)
    graph_path = args.source  + '_dgl_graph.bin'
    mygraph,label_dict = dgl.load_graphs(graph_path)
    mygraph=mygraph[0]
    print("dataset: ", args.source)
    
    n_feats=args.nfeats
    n_classes=args.num_hidden
    f_tensor=th.randn(mygraph.num_nodes(),n_feats)

    l_tensor=th.randint(0,7,(mygraph.num_nodes(),))
    l_tensor=l_tensor.type(th.int64)
    
    testmask=th.ones(mygraph.num_nodes())
    testmask=testmask.type(th.bool)
    trainmask=th.ones(mygraph.num_nodes())
    trainmask=trainmask.type(th.bool)

    mygraph.ndata['label']=l_tensor
    mygraph.ndata['feat']=f_tensor
    mygraph.ndata['test_mask']=testmask
    mygraph.ndata['train_mask']=trainmask
    mygraph.ndata['val_mask']=testmask
    mygraph.ndata['features']=f_tensor
    mygraph.ndata['labels']=l_tensor

    mygraph = dgl.as_heterograph(mygraph)
    train_g = val_g = test_g = mygraph
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    data = n_classes, train_g, val_g, test_g

    train_nfeat = val_nfeat = test_nfeat = mygraph.ndata.pop('features')
    train_labels = val_labels = test_labels = mygraph.ndata.pop('labels')
    test_nid = test_g.ndata.pop('test_mask',
        ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])).nonzero().squeeze()
    train_nid = train_g.ndata.pop('train_mask').nonzero().squeeze()
    val_nid = val_g.ndata.pop('val_mask').nonzero().squeeze()

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    # this to avoid competition overhead on machines with many cores.
    # Change it to a proper number on your machine, especially for multi-GPU training.
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count() // 2 // n_gpus)
    if n_gpus > 1:
        # Copy the graph to shared memory explicitly before pinning.
        # In other cases, we can just rely on fork's copy-on-write.
        # TODO: the original train_g is not freed.
        if args.graph_device == 'uva':
            train_g = train_g.shared_memory('train_g')
        if args.data_device == 'uva':
            train_nfeat = train_nfeat.share_memory_()
            train_labels = train_labels.share_memory_()

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, val_nfeat, test_nfeat, \
           train_labels, val_labels, test_labels, train_nid, val_nid, test_nid
    # my_batch_size=int(int(mygraph.num_nodes())/int(n_gpus))+1

    # my_batch_size = int((mygraph.num_nodes()  + n_gpus - 1)/n_gpus)
    my_batch_size = int(int((mygraph.num_nodes()  + n_gpus - 1)/n_gpus) / 10 * 8)
    # my_batch_size = 1000
    print("my_batch_size: ", my_batch_size)
    
    #print("memory")
    #print(th.cuda.memory_reserved(0))
    #print(th.cuda.memory_allocated(0))
    if devices[0] == -1:
        assert args.graph_device == 'cpu', \
               f"Must have GPUs to enable {args.graph_device} sampling."
        assert args.data_device == 'cpu', \
               f"Must have GPUs to enable {args.data_device} feature storage."
        run(0, 0, args, ['cpu'], data)
    elif n_gpus == 1:
        outfile=open('intermediate0.out','a')
        outfile.write('\n'+str(mygraph.num_nodes())+' '+str(mygraph.num_edges())+' '+str(n_feats)+' ')
        outfile.close()
        run(0, n_gpus, args, devices, data,my_batch_size)
    else:
        procs = []
        for proc_id in range(n_gpus):
            outfile=open('intermediate'+str(proc_id)+'.out','a')
            outfile.write('\n'+str(mygraph.num_nodes())+' '+str(mygraph.num_edges())+' '+str(n_feats)+' ')
            outfile.close()
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data ,my_batch_size))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
