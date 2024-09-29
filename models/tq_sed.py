import torch
import torch.nn as nn
import torchvision

def clip_mse(output, target):

    loss_function = torch.nn.MSELoss(reduction='mean')
    loss = loss_function(output, target)

    return loss


class CRNN(nn.Module):
    def __init__(self, classes_num, cnn_filters, rnn_hid, _dropout_rate, in_channels=1):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(_dropout_rate)

        self.gru1 = nn.GRU(int(3*cnn_filters), rnn_hid, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(rnn_hid*2, rnn_hid)


        self.linear2 = nn.Linear(rnn_hid, classes_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        x = self.conv1(input)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        
        # Bidirectional layer
        recurrent, _ = self.gru1(x)
        x = self.linear1(recurrent)
        x = self.linear2(x)
        return x

class CRNN_LASS_A(nn.Module):
    def __init__(self, classes_num, cnn_filters, rnn_hid, _dropout_rate):
        super(CRNN_LASS_A, self).__init__()
        self.sed_model_list = nn.ModuleList([])
        self.classes_num = classes_num
        for i in range(self.classes_num):
            sed_model = CRNN(1, cnn_filters, rnn_hid, _dropout_rate, 1)
            self.sed_model_list.append(sed_model)
    def forward(self, sep_mel):
        # sep_mel: [batch, classes_num, seq_len, n_mel]
        for i in range(self.classes_num):
            input_mel = sep_mel[:, i, :, :] #[batch, seq_len, n_mel]
            input_mel = input_mel.unsqueeze(1)  #[batch, 1, seq_len, n_mel]
            output = self.sed_model_list[i](input_mel) # [batch, seq_len, 1]
            if i == 0:
                final_out = output
            else:
                final_out = torch.cat((final_out, output), dim=-1)
        return final_out

class CRNN_LASS_B(nn.Module):
    def __init__(self, classes_num, cnn_filters, rnn_hid, _dropout_rate):
        super(CRNN_LASS_B, self).__init__()
        self.sed_model_list = nn.ModuleList([])
        self.classes_num = classes_num
        self.sed_model = CRNN(classes_num, cnn_filters, rnn_hid, _dropout_rate, classes_num)

    def forward(self, sep_mel):
        # sep_mel: [batch, classes_num, seq_len, n_mel]
        output = self.sed_model(sep_mel)

        return output

class CRNN_LASS_C(nn.Module):
    def __init__(self, classes_num, cnn_filters, rnn_hid, _dropout_rate):
        super(CRNN_LASS_C, self).__init__()
        self.sed_model_list = nn.ModuleList([])
        self.classes_num = classes_num
        self.se_block = torchvision.ops.SqueezeExcitation(classes_num, classes_num)
        self.sed_model = CRNN(classes_num, cnn_filters, rnn_hid, _dropout_rate, classes_num)

    def forward(self, sep_mel):
        # sep_mel: [batch, classes_num, seq_len, n_mel]
        x = self.se_block(sep_mel)
        output = self.sed_model(x)

        return output
 
class CRNN_LASS_D(nn.Module):
    def __init__(self, classes_num, cnn_filters, rnn_hid, _dropout_rate):
        super(CRNN_LASS_D, self).__init__()
        self.sed_model_list = nn.ModuleList([])
        self.classes_num = classes_num
        self.attn = nn.MultiheadAttention(embed_dim=11*64, num_heads=8, batch_first=True)
        self.sed_model = CRNN(classes_num, cnn_filters, rnn_hid, _dropout_rate, classes_num)

    def forward(self, sep_mel):
        # sep_mel: [batch, classes_num, seq_len, n_mel]
        batch, class_num, seq_len, n_mel = sep_mel.shape[0], sep_mel.shape[1], sep_mel.shape[2], sep_mel.shape[3]
        x = sep_mel.permute(0, 2, 1, 3)
        x = x.reshape(batch, seq_len, -1)
        x,_ = self.attn(x, x, x)
        x = x.reshape(batch, seq_len, class_num, n_mel)
        x = x.permute(0, 2, 1, 3)
        output = self.sed_model(x)

        return output


if __name__ == '__main__':
    model = CRNN(classes_num=11, cnn_filters=128, rnn_hid=32, _dropout_rate=0.2)
    inputs = torch.rand((2, 1, 200, 64))
    # out = model(inputs)
    # print(out.shape)
    # compute number of parameters
    import numpy as np
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('\nTotal paramters: ' + str(model_params))

    lass_sed_model = CRNN_LASS_A(classes_num=11, cnn_filters=16, rnn_hid=32, _dropout_rate=0.2)
    sep_mel = torch.rand((2, 11, 200, 64))
    # out = lass_sed_model(sep_mel)
    # print(out.shape)
    model_params = sum([np.prod(p.size()) for p in lass_sed_model.parameters()])
    print ('\nTotal paramters: ' + str(model_params))

    lass_sed_model = CRNN_LASS_B(classes_num=11, cnn_filters=128, rnn_hid=32, _dropout_rate=0.2)
    sep_mel = torch.rand((2, 11, 200, 64))
    # out = lass_sed_model(sep_mel)
    # print(out.shape)
    model_params = sum([np.prod(p.size()) for p in lass_sed_model.parameters()])
    print ('\nTotal paramters: ' + str(model_params))

    lass_sed_model = CRNN_LASS_C(classes_num=11, cnn_filters=128, rnn_hid=32, _dropout_rate=0.2)
    sep_mel = torch.rand((2, 11, 200, 64))
    out = lass_sed_model(sep_mel)
    print(out.shape)
    model_params = sum([np.prod(p.size()) for p in lass_sed_model.parameters()])
    print ('\nTotal paramters: ' + str(model_params))

    lass_sed_model = CRNN_LASS_D(classes_num=11, cnn_filters=128, rnn_hid=32, _dropout_rate=0.2)
    sep_mel = torch.rand((2, 11, 200, 64))
    out = lass_sed_model(sep_mel)
    print(out.shape)
    model_params = sum([np.prod(p.size()) for p in lass_sed_model.parameters()])
    print ('\nTotal paramters: ' + str(model_params))