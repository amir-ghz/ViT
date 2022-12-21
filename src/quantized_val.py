import os
import torch
import numpy as np
from tqdm import tqdm
from quantized_model import VisionTransformer
from config import get_eval_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import accuracy, setup_device
from quantization_utils import *
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import random
from resil import eval_model



def main():

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

    data_subset = torch.utils.data.DataLoader(sub_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers)

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

    
    ## Initial quantization based on resiliency
    
    model = unified_weight_quantization(model, 16)

    model.eval()

    base_acc1, base_acc5, base_loss = eval_model(model, data_subset, total_batch, device, config)

    """"

    quant_util = torch.load('quant_cred_resil_to_sensitive.pt')

    with torch.no_grad():
        for layer in quant_util:
            for name, param in model.named_parameters():
                if layer[0][0] == name:
                    if int(layer[1]) < 4:
                        model = mixed_p_weight_quantization(model, layer[0][0], layer[1])

                    if int(layer[1]) >= 4:
                        model = mixed_p_weight_quantization(model, layer[0][0], layer[1])

    model.eval()

    base_acc1, base_acc5, base_loss = eval_model(model, data_subset, total_batch, device, config)

    print(base_acc1)

    """

    resil_list = torch.load('final_resil.pt')

    final_bw = 0

    quant_list = []

    temp_acc = 0

    best_acc = 0

    final_acc = 0

    is_quantized = False # shows whethere a layer can be quantized initially or not

    with torch.no_grad():
        for layer in resil_list:
                for name, param in model.named_parameters():
                    if name == layer[0]:
                        for bw in reversed(range(2, 16)):

                            if bw < 4:

                                model = binary_ternary_quantization(model, name, bw)

                                temp_acc, base_acc5, base_loss = eval_model(model, data_subset, total_batch, device, config)

                            else:

                                model = mixed_p_weight_quantization(model, name, bw)

                                temp_acc, base_acc5, base_loss = eval_model(model, data_subset, total_batch, device, config)

                            if temp_acc >= 97.80:

                                torch.save(model.state_dict(), 'Q_tran.pt')
                                final_bw = bw
                                best_acc = temp_acc
                                is_quantized = True
                                final_acc = temp_acc

                            else:
                                final_bw = bw + 1
                                break
                        
                model.load_state_dict(torch.load('Q_tran.pt'))
                model.eval()
                quant_list.append([layer, final_bw, best_acc])
                print('acc diff is: ', base_acc1 - best_acc, ' and layer', layer, ' is quantized by: ', final_bw, ' bit-width')
                best_acc = 1000

        torch.save(quant_list, 'quant_cred.pt')
        print('final acc is: ', final_acc)


    #base_acc1, base_acc5, base_loss = eval_model(model, data_subset, total_batch, device, config)



if __name__ == '__main__':
    main()