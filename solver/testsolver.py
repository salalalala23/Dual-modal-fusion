from solver.basesolver import BaseSolver
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import numpy as np
from PIL import Image

class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module("model." + net_name)
        net = lib.Net
        
        