from visdom import Visdom
from config import config
from dataloader import get_data
import torch as t
from model import PoetryModel
from torch import nn
import tqdm
from torch.autograd import Variable as V 
import ipdb 

opt = config()
def train(**kwargs:dict) -> None:

    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    #vis = Visdom(env=opt.env)

    #data get 
    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(
        data, 
        batch_size=opt.batch_size,
        shuffle = True,   
        )
    
    model = PoetryModel(len(word2ix), 2, 2)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.user_gpu:
        model.cuda()
        criterion.cuda()
    
    for epoch in range(opt.epoch):
        
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            data_ = data.long().transpose(1, 0).contiguous()
            if opt.user_gpu : data_ = data_.cuda()
            optimizer.zero_grad()
            input_, target = V(data_[:-1, :]), V(data[1:, :])
            ouput, _ = model(input_)
            loss = criterion(ouput, target.view(-1))
            loss.backward()
            optimizer.step()

            #if (1 + ii)%opt.plot_every == 0:
            #    if os.path.exists(opt.debug_file):
            #        ipdb.set_trace()
            
            #    poetrys = \
            #        [[ix2word[_word] for _word in data_[:, iii]]
            #            for _iii in range(data_.size(1))][:16]

            #    vis.text("</br>".join([" ".join(poetry) 
            #            for poetry in poetrys]), 
            #            win="origin_poem")
            
            #    gen_poetries = []
                
            #    for word in list():
            #        gen_poetry = " ".join(generate(model, word, ix2word, word2ix))
            #        gen_poetries.append(gen_poetry)
                
                #vis.text("</br>".join(''.join(poetry) for poetry in gen_poetries]),
                #    win="gen_poem")
                
    #t.save(model.state_dict(), "%s_%s.pth" %(opt.model_prefix, epoch))

train()                  





    
