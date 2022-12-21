import torch


def unified_weight_quantization(model, bit_width):

    with torch.no_grad():
        for name, param in model.named_parameters():
            values = param.data

            max  = torch.max(values)
            min = torch.min(values)
            scaling_fac = (max-min)/((2**bit_width)-1)
            param.data = torch.mul(torch.floor(torch.div(values, scaling_fac)), scaling_fac)
    
    model.eval()

    return model

def mixed_p_weight_quantization(model, layer_name, bit_width):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if layer_name == name:
                
                values = param.data
                max  = torch.max(values)
                min = torch.min(values)
                scaling_fac = (max-min)/((2**bit_width)-1)
                param.data = torch.mul(torch.floor(torch.div(values, scaling_fac)), scaling_fac)
    
    model.eval()

    return model


def binary_ternary_quantization(model, layer_name: str, bit_width: int):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if layer_name == name:
                
                values = param.data

                if bit_width == 2:
                    for i in range(len(values.view(-1))):
                        if values.view(-1)[i] > 0:
                            values.view(-1)[i] = 1
                        else: 
                            values.view(-1)[i] = 0


                if bit_width == 3:
                    for i in range(len(values.view(-1))):
                        if values.view(-1)[i] > 0:
                            values.view(-1)[i] = 1
                        if values.view(-1)[i] == 0: 
                            values.view(-1)[i] = 0
                        else:
                            values.view(-1)[i] = -1

                    
                param.data = values
    
    model.eval()

    return model


def output_quantization(tensor, bit_width):

    values = tensor
    max  = torch.max(values)
    min = torch.min(values)
    scaling_fac = (max-min)/((2**bit_width)-1)
    output_tensor = torch.mul(torch.floor(torch.div(values, scaling_fac)), scaling_fac)

    return output_tensor