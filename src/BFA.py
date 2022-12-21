import os
import torch
import numpy as np
from tqdm import tqdm
from quantized_model import VisionTransformer
from config import get_eval_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import accuracy, setup_device
from quantization_utils import unified_weight_quantization
from binary_utils import *
import random
import math
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, SVHN, GTSRB
import copy


def random_faullt_injector(model, quantized_model, q_bw, min_bf_count, max_bf_count, bf_step):

    """
    model              --> the full precision model
    quantized_model    --> the quantized model
    q_bw               --> the INT quantization bit width
    bf_count           --> the number of bit flips

    """

    # Loading subset of data:

    config = get_eval_config()

    device, device_ids = setup_device(config.n_gpu)

    data_loader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val')
    
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    data_loader =  datasets.CIFAR10('./data/CIFAR10', train=False, transform=transform, download=True)

    random_idx = []

    for i in range(0, 1000):
        random_idx.append(random.randint(0,9999))
        
    sub_dataset = Subset(data_loader, random_idx)

    data_subset = torch.utils.data.DataLoader(sub_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)

    #total_batch = len(data_loader)
    total_batch = len(data_subset)


    temp_max_bf_count = max_bf_count

    model.eval()
    quantized_model.eval()

    layers_scaling_fac = []

    with torch.no_grad():
        
        for name, param in model.named_parameters():
            values = param.data

            max  = torch.max(values)
            min = torch.min(values)
            scaling_fac = (max-min)/((2**q_bw)-1)
            layers_scaling_fac.append([name, scaling_fac])

    analysis_list = []

    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            weight_tensor = param.data
            
            scaling_fac = 0

            for item in layers_scaling_fac:
                if(item[0] == name):
                    scaling_fac = item[1]


            if(torch.numel(weight_tensor)*q_bw < max_bf_count):
                max_bf_count = torch.numel(weight_tensor)*q_bw - (2*bf_step)
            else:
                max_bf_count = temp_max_bf_count


            for i in range(min_bf_count, max_bf_count, bf_step):

                temp_model = quantized_model
                temp_model.eval()

                for j in range(0, i):

                    random_idx = random.randint(0, len(weight_tensor)-1)

                    target_weight = weight_tensor.view(-1,)[random_idx] 

                    target_weight_binary = int_to_binary(int((target_weight)/scaling_fac), q_bw)

                    binary_faulty_weight = bit_flip(target_weight_binary, random.randint(0, q_bw-1)) #random.randint(0, 7)

                    faulty_weight = binary_to_int(binary_faulty_weight)

                    for temp_name, parameter in temp_model.named_parameters():
                        with torch.no_grad():
                            if(temp_name == name):
                                parameter.data.view(-1,)[random_idx] = faulty_weight*scaling_fac
                
                temp_model.eval()

                analysis_list.append([name, max_bf_count, i, eval_model(temp_model, data_subset, total_batch, device, config)])

                print(analysis_list)

        torch.save(analysis_list, 'results.pt')


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
    return [np.mean(acc1s), np.mean(acc5s), sum(loss) / len(loss)]

def main():

    config = get_eval_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

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


    # for test

    data_loader = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val')
    
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    data_loader =  datasets.CIFAR10('./data/CIFAR10', train=False, transform=transform, download=True)

    random.seed(10)
    random_idx = []
    for i in range(0, 1000):
        random_idx.append(random.randint(0,9999))
        
    sub_dataset = Subset(data_loader, random_idx)

    data_subset = torch.utils.data.DataLoader(sub_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers)

    #total_batch = len(data_loader)
    total_batch = len(data_subset)

    eval_model(model, data_subset, total_batch, device, config)

    # create dataloader
    data_loader_train = eval("{}DataLoader".format(config.dataset))(
                    data_dir=os.path.join(config.data_dir, config.dataset),
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='train')




    # quantized all model weights to INT8:

    quantized_model = unified_weight_quantization(model, 8)


    # accumulate gradient:

    iterations = iter(data_loader_train)
    quantized_model.train()
    data = next(iterations)
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = quantized_model(inputs)

    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

    





    quantized_model.eval()
    eval_model(quantized_model, data_subset, total_batch, device, config)

if __name__ == '__main__':
    main()