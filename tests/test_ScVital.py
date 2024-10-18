import scVital as scVt
import torch
import numpy as np 
import pandas as pd 
import scanpy as sc 

def testHelloWorld():
    assert scVt.helloWorld() == "Hello World"

def testMinusOne():
    assert scVt.minusOne(4) == 3
