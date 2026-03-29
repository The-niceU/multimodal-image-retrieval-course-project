import logging
import os
import torch
import time
import sys
import random
import numpy as np
import logging
from torch.utils.data import dataloader
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast, GradScaler

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def get_log(dataset_name):
    if not os.path.exists('log/'):
        os.mkdir("log/")
    if not os.path.exists('./log/' + dataset_name + '/'):
        os.mkdir('./log/' + dataset_name + '/')
    timestamp = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
    log_folder_path = os.path.join("./log/" + dataset_name + '/' + timestamp)
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)
    log_format = '%(asctime)s: %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    log_file_path = os.path.join(log_folder_path, timestamp + '.log')
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return log_folder_path

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval() 

def train_and_evaluate(model, optimizer, trainset, args, device):
    trainloader = dataloader.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    scaler = GradScaler()
    model.train()
    model.apply(set_bn_eval)

    for epoch in range(args.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        loss_avg = RunningAverage()

        with tqdm(total=len(trainloader)) as t:
            for i, data in enumerate(trainloader):
                reference_image = data['source_img_data'].to(device)
                target_image = data['target_img_data'].to(device)
                mods = data['mod']['str']
                optimizer.zero_grad()

                with autocast():
                    loss = model.compute_loss(reference_image, mods, target_image)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_avg.update(loss.item())
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

        test(args, model, trainset, device)


def test(params, model, testset, device):
    model.eval()
   
    test_queries = testset.test_queries
    test_targets = testset.test_targets

    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            # compute test query features
            imgs = []
            mods = []
            for t in test_queries:
                imgs += [t['source_img_data']]
                mods += [t['mod']['str']]
                if len(imgs) >= params.batch_size or t is test_queries[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().to(device)
                    f = model.extract_query(mods, imgs)
                    f = f.data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    mods = []
            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            for t in test_targets:
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().to(device)
                    imgs = model.extract_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
    
    
    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

    # compute recalls
    out = []
    for k in [1, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        out.append(round(r,3))

    logging.info(f'R@1: {out[0]}, R@10: {out[1]}, R@50: {out[2]}, Avg(R@10,R@50): {round((out[1]+out[2])/2,3)}')