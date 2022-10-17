import jittor as jt
import argparse

from numpy import dtype

parser = argparse.ArgumentParser()
parser.add_argument("--test",action="store_true")
parser.add_argument("--iter",type=int,default= 10000)




args = parser.parse_args()

if(args.test):
    print("Test!!")
    
else:
    print(f"Train! Iters = {args.iter}")
