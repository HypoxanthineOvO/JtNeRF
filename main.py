import jittor as jt
import argparse
from utils import Trainer

if jt.has_cuda:
    print("Use CUDA !!!")
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description= "Jittor NeRF Framework!!")
parser.add_argument("--test",action="store_true")

parser.add_argument("--iter",type=int,default= 10000)

parser.add_argument("--data_type",default='npz')
parser.add_argument("--dir",default='./data/tiny_nerf_data.npz')
parser.add_argument("--batchsize",type= int,default = 1)

parser.add_argument("--enc",default= "pos")
parser.add_argument("--encpara",type=int,default= 6)



args = parser.parse_args()

print(args)

data = {'type':args.data_type,'root_dir':args.dir,'batch_size':args.batchsize}
encoder = {'type':args.enc,'para':args.encpara}


if(args.test):
    print("Test!!")
    print("But 我还没写, tnnd, 开摆")
    
else:
    trainer = Trainer('Trial',data,encoder)
    trainer.train()
