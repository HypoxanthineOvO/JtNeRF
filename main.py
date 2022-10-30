import jittor as jt
import argparse
from runhelper import Trainer

if jt.has_cuda:
    print("Use CUDA !!!")
    jt.flags.use_cuda = 1

# Status
parser = argparse.ArgumentParser(description= "Jittor-NeRF-Framework")
parser.add_argument("--test",action="store_true")
parser.add_argument("--output",default="./out/test/")

# Basic parameters
parser.add_argument("--lr",type = float,default= 1e-5)
parser.add_argument("--iter",type=int,default= 1000)

# About load data
parser.add_argument("--model",default="./out/test/test_model.pkl") # Jittor only support pkl
parser.add_argument("--data_type",default='npz')
parser.add_argument("--dir",default='./data/tiny_nerf_data.npz')
parser.add_argument("--batchsize",type= int,default = 2)

# About Encoding
parser.add_argument("--enc",default= "pos")
parser.add_argument("--encpara",type=int,default= 6)



args = parser.parse_args()

print(args)

data = {'type':args.data_type,'root_dir':args.dir,'batch_size':args.batchsize,'out':args.output}
encoder = {'type':args.enc,'para':args.encpara}
train_parameters = {'lr':args.lr,"iters":args.iter}


if(args.test):
    print("Test!!")
    tester = Trainer("Test",data,encoder,train_parameters,model = args.model)
    
else:
    if args.model != "None":
        trainer = Trainer('Trial',data,encoder,train_parameters,model=args.model)
    else:
        print(train_parameters)
        trainer = Trainer('Trial',data,encoder,train_parameters)
    trainer.train()
