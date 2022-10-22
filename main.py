import jittor as jt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--test",action="store_true")
parser.add_argument("--iter",type=int,default= 10000)
parser.add_argument("--enc",default= "posenc")
parser.add_argument("--encpara",type=int,default= 6)



args = parser.parse_args()

if(args.test):
    print("Test!!")
    print("But 我还没写, tnnd, 开摆")
    
else:
    print(f"Train! Iters = {args.iter}")
