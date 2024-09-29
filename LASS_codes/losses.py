import torch
import torch.nn as nn

def l1(output, target):
    return torch.mean(torch.abs(output - target))

def smoothl1_wav(output_dict, target_dict):
    loss_fun = nn.SmoothL1Loss()
    return loss_fun(output_dict['segment'], target_dict['segment'])

def l1_wav(output_dict, target_dict):
	return l1(output_dict['segment'], target_dict['segment'])


def spec_loss(output_dict, target_dict):
    output = output_dict['segment']
    target = target_dict['segment']
    # print(output.shape, target.shape)
    if output.ndim != 2 and output.ndim != 1:
        print('Reshape out ...')
        output = output.reshape(output.shape[0], -1)
    if target.ndim != 2 and target.ndim != 1:
        print('Reshape out ...')
        target = target.reshape(target.shape[0], -1)
 
    loss_func = nn.SmoothL1Loss()
    output_spec = torch.stft(output, n_fft=2048, hop_length=320, win_length=2048,
                             return_complex=(True))
    target_spec = torch.stft(target, n_fft=2048, hop_length=320, win_length=2048,
                             return_complex=(True))
    mag_loss1 = loss_func(torch.real(output_spec), torch.real(target_spec))
    mag_loss2 = loss_func(torch.imag(output_spec), torch.imag(target_spec))
    mag_loss = mag_loss1 + mag_loss2
    
    return mag_loss        

def wav_spec(output_dict, target_dict):
    loss_wav = l1_wav(output_dict, target_dict)
    loss_spec = spec_loss(output_dict, target_dict)
    
    return loss_wav + loss_spec * 0.05


def get_loss_function(loss_type):
    if loss_type == "l1_wav":
        return l1_wav
    elif loss_type == "smoothl1_wav":
        return smoothl1_wav
    elif loss_type == 'spec':
        return spec_loss
    elif loss_type == 'l1_wav&spec':
        return wav_spec
    else:
        raise NotImplementedError("Error!")


if __name__ == '__main__':
    audio = torch.rand((1,1,160000))
    output_dict = {'segment': audio}
    # out = l1_spec(output_dict, output_dict)