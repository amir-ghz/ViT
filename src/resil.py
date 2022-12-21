import os
import torch
import numpy as np
from tqdm import tqdm
from model import VisionTransformer
from config import get_eval_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import accuracy, setup_device, float2bit, bit2float
from quantization_utils import unified_weight_quantization
from binary_utils import *
import random
import math
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, SVHN, GTSRB
import copy



def eval_model(model, data_subset, total_batch, device, config):

    # starting evaluation
    print("Starting evaluation")
    acc1s = []
    acc5s = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(data_subset), total=total_batch)
        loss = []
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))

            data = data.to(device)
            target = target.to(device)

            pred_logits = model(data)
            acc1, acc5 = accuracy(pred_logits, target, topk=(1, 5))

            loss.append(torch.nn.CrossEntropyLoss()(pred_logits, target).item())

            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

            pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item())

    print("Evaluation of model {:s} on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}, loss: {:.4f}".format(config.model_arch, config.dataset, np.mean(acc1s), np.mean(acc5s), sum(loss) / len(loss)))
    return np.mean(acc1s), np.mean(acc5s), sum(loss) / len(loss)

def main():
    
    # Loading subset of data:

    config = get_eval_config()

    device, device_ids = setup_device(config.n_gpu)
    
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    data_loader =  datasets.CIFAR10('../data/dummyEx/CIFAR10/', train=False, transform=transform, download=True)

    random_idx = torch.load('random.pt')
        
    sub_dataset = Subset(data_loader, random_idx)

    data_subset = torch.utils.data.DataLoader(sub_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)

    #total_batch = len(data_loader)
    total_batch = len(data_subset)


    # create model
    model = VisionTransformer(
             image_size=(config.image_size, config.image_size),
             patch_size=(config.patch_size, config.patch_size),
             emb_dim=config.emb_dim,
             mlp_dim=config.mlp_dim,
             num_heads=config.num_heads,
             num_layers=config.num_layers,
             num_classes=config.num_classes,
             attn_dropout_rate=config.attn_dropout_rate,
             dropout_rate=config.dropout_rate)


    # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        model.load_state_dict(state_dict)
        #model.load_state_dict(torch.load(config.checkpoint_path).state_dict())
        print("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)





    model.eval()
    base_acc1, base_acc5, base_loss = eval_model(model, data_subset, total_batch, device, config)


    print("the base accuracy is: ", base_acc1, " and the loss is: ", base_loss)


    BER = [10, 50, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000, 20_000, 50_000, 100_000]
    res_list = []




    par_result = torch.load("./par_result_0.pt")
    par_result_2 = torch.load("./par_result_1.pt")
    par_result.extend(par_result_2)

    layer_names = []
    for i in range(0, len(par_result)):
        layer_names.append(par_result[i][0])

    
    count = 0

    for name, param in model.named_parameters(): 
        count += 1

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@: ", count)


    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in layer_names:
                model = model_refresher(config, model)

                for ber in BER:

                    model_refresher(config, model)

                    target_layer = param.data.to('cpu')

                    binary_tensor = float2bit(target_layer, num_e_bits=8, num_m_bits=23, bias=127.).to(device)

                    if(ber >= (torch.numel(binary_tensor))-1):
                        break

                    for i in range(0, ber):
                        idx = random.randint(0, (torch.numel(binary_tensor))-1)
                        
                        if binary_tensor.view(-1,)[idx] == float(0):
                            binary_tensor.view(-1,)[idx] = float(1)
                        else:
                            binary_tensor.view(-1,)[idx] = float(0)


                    float_tensor = bit2float(binary_tensor, num_e_bits=8, num_m_bits=23, bias=127.).to(device)

                    with torch.no_grad():
                        param.data = float_tensor

                    model.eval()

                    faulty_acc1, faulty_acc5, faulty_loss = eval_model(model, data_subset, total_batch, device, config)

                    print("the faulty accuracy is: ", faulty_acc1, " and the loss is: ", faulty_loss)

                    res_list.append([name, faulty_acc1, faulty_acc5, faulty_loss, ber])

                    torch.save(res_list, 'par_result.pt')


    torch.save(res_list, 'result.pt')

def model_refresher(config, model):

    state_dict = load_checkpoint(config.checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    return model

if __name__ == '__main__':
    main()

        






