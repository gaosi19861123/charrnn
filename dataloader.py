import numpy as np
import torch as t
from config import config
import os 
opt = config()

def get_data(opt):

    if os.path.exists(opt.pickle_path):
        datas = np.load(opt.pickle_path, allow_pickle=True)
        data, word2ix, ix2word = datas["data"], datas["word2ix"].item(), datas["ix2word"].item() 
        return data, word2ix, ix2word




