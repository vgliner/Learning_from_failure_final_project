import torch
import torch.nn as nn
import math
import time
from torch.autograd import Function
import numpy as np

# Autograd Function objects are what record operation history on tensors,
# and define formulas for the forward and backprop.

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha = 0.0):
        # Store context for backprop
        ctx.alpha = alpha
        
        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        # print(f'{ctx.alpha}')
        output = grad_output.neg() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None



# A simple but versatile d1 convolutional neural net
class ConvNet1d(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_lengths: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv1d(layer_in_channels, layer_out_channels, kernel_size=kernel_lengths[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# A simple but versatile d2 convolutional neural net
class ConvNet2d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_sizes: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        self.dual_kernel = False
        if np.shape(kernel_sizes)[0]==2:
            self.dual_kernel = True
        if self.dual_kernel == False:
            assert len(hidden_channels) == len(kernel_sizes)
        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            if self.dual_kernel:
                kernel_size= [kernel_sizes[0][i],kernel_sizes[1][i]]
            else:
                kernel_size= kernel_sizes[i]
            layers.append(nn.Conv2d(layer_in_channels, layer_out_channels, kernel_size=kernel_size,
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm2d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# A simple but versatile d1 "deconvolution" neural net
class DeConvNet1d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, out_kernel: int,
                 kernel_lengths: list, dropout=None, stride=1, dilation=1, batch_norm=False, output_padding=1):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.ConvTranspose1d(layer_in_channels, layer_out_channels, kernel_size=kernel_lengths[i],
                                             stride=stride, dilation=dilation, output_padding=output_padding))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2))

            layer_in_channels = layer_out_channels

        layers.append(nn.ConvTranspose1d(layer_in_channels, out_channels, out_kernel, stride, dilation))

        self.dcnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dcnn(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# A simple but versatile d2 convolutional neural net with skipping connections
class ConvNet2d_with_Skip(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_sizes: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):
            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv2d(layer_in_channels, layer_out_channels, kernel_size=kernel_sizes[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm2d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            layer_in_channels = layer_out_channels
        self.cnn = nn.Sequential(*layers)      

    def forward(self, x):
        return self.cnn(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Ecg12LeadNet(nn.Module):
    def forward(self, x):
        x1, x2 = x
        out1 = self.short_cnn(x1).reshape((x1.shape[0], -1))
        out2 = self.long_cnn(x2).reshape((x2.shape[0], -1))
        out = torch.cat((out1, out2), 1)
        return self.fc(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features        

    def __init__(self,
                 short_hidden_channels: list, long_hidden_channels: list,
                 short_kernel_lengths: list, long_kernel_lengths: list,
                 fc_hidden_dims: list,
                 short_dropout=None, long_dropout=None,
                 short_stride=1, long_stride=1,
                 short_dilation=1, long_dilation=1,
                 short_batch_norm=False, long_batch_norm=False,
                 short_input_length=1250, long_input_length=5000,
                 num_of_classes=2):
        super().__init__()
        assert len(short_hidden_channels) == len(short_kernel_lengths)
        assert len(long_hidden_channels) == len(long_kernel_lengths)

        self.short_cnn = ConvNet1d(12, short_hidden_channels, short_kernel_lengths, short_dropout,
                                   short_stride, short_dilation, short_batch_norm)
        self.long_cnn = ConvNet1d(1, long_hidden_channels, long_kernel_lengths, long_dropout,
                                  long_stride, long_dilation, long_batch_norm)

        short_out_channels = short_hidden_channels[-1]
        short_out_dim = short_out_channels * calc_out_length(short_input_length, short_kernel_lengths,
                                                             short_stride, short_dilation)
        long_out_channels = long_hidden_channels[-1]
        long_out_dim = long_out_channels * calc_out_length(long_input_length, long_kernel_lengths,
                                                           long_stride, long_dilation)

        in_dim = short_out_dim + long_out_dim
        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)

class Ecg12LeadMultiClassNet(nn.Module):
    def forward(self, x):
        x1, x2 = x
        out1 = self.short_cnn(x1).reshape((x1.shape[0], -1))
        out2 = self.long_cnn(x2).reshape((x2.shape[0], -1))
        out = torch.cat((out1, out2), 1)
        return self.fc(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features        

    def __init__(self,
                 short_hidden_channels: list, long_hidden_channels: list,
                 short_kernel_lengths: list, long_kernel_lengths: list,
                 fc_hidden_dims: list,
                 short_dropout=None, long_dropout=None,
                 short_stride=1, long_stride=1,
                 short_dilation=1, long_dilation=1,
                 short_batch_norm=False, long_batch_norm=False,
                 short_input_length=1250, long_input_length=5000,
                 num_of_classes=2):
        super().__init__()
        assert len(short_hidden_channels) == len(short_kernel_lengths)
        assert len(long_hidden_channels) == len(long_kernel_lengths)

        self.short_cnn = ConvNet1d(12, short_hidden_channels, short_kernel_lengths, short_dropout,
                                   short_stride, short_dilation, short_batch_norm)
        self.long_cnn = ConvNet1d(1, long_hidden_channels, long_kernel_lengths, long_dropout,
                                  long_stride, long_dilation, long_batch_norm)

        short_out_channels = short_hidden_channels[-1]
        short_out_dim = short_out_channels * calc_out_length(short_input_length, short_kernel_lengths,
                                                             short_stride, short_dilation)
        long_out_channels = long_hidden_channels[-1]
        long_out_dim = long_out_channels * calc_out_length(long_input_length, long_kernel_lengths,
                                                           long_stride, long_dilation)

        in_dim = short_out_dim + long_out_dim
        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)


class Ecg12ImageNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, kernel_sizes: list, in_h: int, in_w: int,
                 fc_hidden_dims: list, dropout=None, stride=1, dilation=1, batch_norm=False, num_of_classes=2):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        self.cnn = ConvNet2d(in_channels, hidden_channels, kernel_sizes, dropout, stride, dilation, batch_norm)

        out_channels = hidden_channels[-1]
        out_h = calc_out_length(in_h, kernel_sizes, stride, dilation)
        out_w = calc_out_length(in_w, kernel_sizes, stride, dilation)
        in_dim = out_channels * out_h * out_w

        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape((x.shape[0], -1))
        return self.fc(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Ecg12ImageToSignalNet(nn.Module):
    def __init__(self, in_channels: int, deconv_in_channels: int, in_h: int, in_w: int,
                 conv_hidden_channels: list, conv_kernel_sizes: list,
                 deconv_hidden_channels_short: list, deconv_kernel_lengths_short: list,
                 deconv_hidden_channels_long: list, deconv_kernel_lengths_long: list,
                 conv_dropout=None, conv_stride=1, conv_dilation=1, conv_batch_norm=False,
                 deconv_dropout_short=None, deconv_stride_short=1, deconv_dilation_short=1,
                 deconv_batch_norm_short=False, deconv_out_kernel_short=5,
                 deconv_dropout_long=None, deconv_stride_long=1, deconv_dilation_long=1,
                 deconv_batch_norm_long=False, deconv_out_kernel_long=5,
                 fc_hidden_dims=(), l_out_long=5000, l_out_short=1250, short_leads=12, long_leads=1):
        super().__init__()

        self.deconv_in_channels = deconv_in_channels
        self.l_out_short = l_out_short
        self.l_out_long = l_out_long

        self.l_in_short = calc_out_length(l_out_short, deconv_kernel_lengths_short + [deconv_out_kernel_short],
                                          deconv_stride_short, deconv_dilation_short)
        self.l_in_long = calc_out_length(l_out_long, deconv_kernel_lengths_long + [deconv_out_kernel_long],
                                         deconv_stride_long, deconv_dilation_long)

        self.short_leads_latent_dim = self.l_in_short*deconv_in_channels
        self.long_leads_latent_dim = self.l_in_long*deconv_in_channels
        latent_dim = self.short_leads_latent_dim + self.long_leads_latent_dim

        self.cnn2d = Ecg12ImageNet(in_channels, hidden_channels=conv_hidden_channels, kernel_sizes=conv_kernel_sizes,
                                   in_h=in_h, in_w=in_w, fc_hidden_dims=fc_hidden_dims, dropout=conv_dropout,
                                   stride=conv_stride, dilation=conv_dilation, batch_norm=conv_batch_norm,
                                   num_of_classes=latent_dim)
        self.ReLu = nn.ReLU()
        self.dcnn1d_short = DeConvNet1d(deconv_in_channels, hidden_channels=deconv_hidden_channels_short,
                                        out_channels=short_leads, out_kernel=deconv_out_kernel_short,
                                        kernel_lengths=deconv_kernel_lengths_short,
                                        dropout=deconv_dropout_short, stride=deconv_stride_short,
                                        dilation=deconv_dilation_short, batch_norm=deconv_batch_norm_short)
        self.dcnn1d_long = DeConvNet1d(deconv_in_channels, hidden_channels=deconv_hidden_channels_long,
                                       out_channels=long_leads, out_kernel=deconv_out_kernel_long,
                                       kernel_lengths=deconv_kernel_lengths_long,
                                       dropout=deconv_dropout_long, stride=deconv_stride_long,
                                       dilation=deconv_dilation_long, batch_norm=deconv_batch_norm_long)

    def forward(self, x):
        out = self.cnn2d(x)
        out = self.ReLu(out)

        latent_long = out[:, 0:self.long_leads_latent_dim]
        latent_long = latent_long.reshape((x.shape[0], self.deconv_in_channels, self.l_in_long))
        latent_short = out[:, self.long_leads_latent_dim:self.long_leads_latent_dim + self.short_leads_latent_dim]
        latent_short = latent_short.reshape((x.shape[0], self.deconv_in_channels, self.l_in_short))

        out_short = self.dcnn1d_short(latent_short)
        start_short = (out_short.shape[2] - self.l_out_short) // 2
        out_long = self.dcnn1d_long(latent_long)
        start_long = (out_long.shape[2] - self.l_out_long) // 2
        return out_short[:, :, start_short:start_short + self.l_out_short], \
            out_long[:, :, start_long:start_long + self.l_out_long]


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NewNet(nn.Module):

    def __init__(self, in_channels: int, in_h: int, in_w: int,
                 conv_hidden_channels: list, conv_kernel_sizes: list,
                 conv_dropout=None, conv_stride=1, conv_dilation=1, conv_batch_norm=False,
                 fc_hidden_dims=(), l_out_long=5000, l_out_short=1250, short_leads=12, long_leads=1):

        super().__init__()
        self.short_leads = short_leads
        self.long_leads = long_leads
        self.l_out_short = l_out_short
        self.l_out_long = l_out_long
        self.out_dim_short = l_out_short*short_leads
        self.out_dim_long = l_out_long*long_leads
        self.out_dim = self.out_dim_long + self.out_dim_short

        self.cnn2d = Ecg12ImageNet(in_channels, hidden_channels=conv_hidden_channels, kernel_sizes=conv_kernel_sizes,
                                   in_h=in_h, in_w=in_w, fc_hidden_dims=fc_hidden_dims, dropout=conv_dropout,
                                   stride=conv_stride, dilation=conv_dilation, batch_norm=conv_batch_norm,
                                   num_of_classes=self.out_dim)

        self.cnn1d_short = nn.Conv1d(short_leads, short_leads, 5, 1, 2, 1, 1, True)
        self.cnn1d_long = nn.Conv1d(long_leads, long_leads, 5, 1, 2, 1, 1, True)

        self.s = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.cnn2d(x)
        out_long = out[:, 0:self.out_dim_long].reshape(batch_size, self.long_leads, self.l_out_long)
        out_short = out[:, self.out_dim_long:].reshape(batch_size, self.short_leads, self.l_out_short)
        out_long = self.cnn1d_long(out_long)
        out_short = self.cnn1d_short(out_short)
        return self.s(out_short), self.s(out_long)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def calc_out_length(l_in: int, kernel_lengths: list, stride: int, dilation: int, padding=0):
    l_out = l_in
    for kernel in kernel_lengths:
        l_out = math.floor((l_out + 2*padding - dilation * (kernel - 1) - 1) / stride + 1)
    return l_out


def calc_out_length_deconv(l_in: int, kernel_lengths: list, out_kernel: int, stride: int, dilation: int,
                           padding=0, output_padding=0):
    l_out = l_in
    for kernel in kernel_lengths:
        l_out = (l_out - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1
    l_out = (l_out - 1)*stride - 2*padding + dilation*(out_kernel - 1) + 1
    return l_out



class Image_Classifier_inc_Domain_Adaptation(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, kernel_sizes: list, in_h: int, in_w: int,
                 fc_hidden_dims: list, dropout=None, stride=1, dilation=1, batch_norm=False, num_of_classes=2,grl_lambda=1.0, skip_connections = None):
        super().__init__()
        if np.shape(kernel_sizes)[0] == 2:
            assert len(hidden_channels) == len(kernel_sizes[0])
        else:
            assert len(hidden_channels) == len(kernel_sizes)
        self.skip_connections = skip_connections
        #####   Feature extractor   ########
        self.feature_extractor  = ConvNet2d(in_channels, hidden_channels, kernel_sizes, dropout, stride, dilation, batch_norm)
        if skip_connections != None:
            self.feature_extractor_with_skip  = ConvNet2d(in_channels, hidden_channels[:skip_connections+1], kernel_sizes[:skip_connections+1], dropout, stride, dilation, batch_norm)

        #####   END of Feature extractor   ########
        #####   Class Classifier   ########


        out_channels = hidden_channels[-1]
        if np.shape(kernel_sizes)[0]==2:
            out_h = calc_out_length(in_h, kernel_sizes[0], stride, dilation)
            out_w = calc_out_length(in_w, kernel_sizes[1], stride, dilation)
        else:
            out_h = calc_out_length(in_h, kernel_sizes, stride, dilation)
            out_w = calc_out_length(in_w, kernel_sizes, stride, dilation)
        in_dim = out_channels * out_h * out_w

        if skip_connections != None:
            out_h_skip = calc_out_length(in_h, kernel_sizes[:skip_connections+1], stride, dilation)
            out_w_skip = calc_out_length(in_w, kernel_sizes[:skip_connections+1], stride, dilation)            
            in_dim_skip = hidden_channels[skip_connections] * out_h_skip * out_w_skip
            skip_conn_adjust_layers =[]        
            skip_conn_adjust_layers.append(nn.Linear(in_dim_skip, in_dim))
            skip_conn_adjust_layers.append(nn.ReLU())


        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        layers2 = []
        in_dim = out_channels * out_h * out_w
        for out_dim in fc_hidden_dims:
            layers2.append(nn.Linear(in_dim, out_dim))
            layers2.append(nn.ReLU())
            in_dim = out_dim


        # single score for binary classification, class score for multi-class
        layers.append(nn.Linear(in_dim, 1))
        layers2.append(nn.Linear(in_dim, 1))

        self.class_classifier = nn.Sequential(*layers)
        self.domain_classifier = nn.Sequential(*layers2)
        if skip_connections != None:
            self.class_classifier_adj_skip = nn.Sequential(*skip_conn_adjust_layers)
            self.domain_classifier_adj_skip = nn.Sequential(*skip_conn_adjust_layers)            

        self.grl_lambda = grl_lambda


    def forward(self, x):        
        features = self.feature_extractor(x)
        reverse_features = GradientReversalFn.apply(features, self.grl_lambda) 
        # if self.skip_connections != None:
        #     features_with_skip = self.feature_extractor_with_skip(x)
        #     reverse_features_with_skip = GradientReversalFn.apply(features_with_skip, self.grl_lambda) 

        out_features = features.reshape((x.shape[0], -1))     
        out_reverse_features = reverse_features.reshape((x.shape[0], -1))   
        
        # if self.skip_connections != None:
        #     out_features_with_skip = features_with_skip.reshape((x.shape[0], -1))     
        #     out_reverse_features_with_skip = reverse_features_with_skip.reshape((x.shape[0], -1))
        #     out_features_with_skip_adj = self.class_classifier_adj_skip(out_features_with_skip)
        #     out_reverse_features_with_skip_adj = self.domain_classifier_adj_skip(out_reverse_features_with_skip)
        #     class_pred = self.class_classifier(out_features+out_features_with_skip_adj)
        #     domain_pred = self.domain_classifier(out_reverse_features+out_reverse_features_with_skip_adj)
        # else:
        class_pred = self.class_classifier(out_features)
        domain_pred = self.domain_classifier(out_reverse_features)            
        return class_pred, domain_pred