import sys
# sys.path.append('/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes')

import numpy as np
from typing import Dict, List, NoReturn, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from .base import Base, init_layer, init_bn, act
from .FaSNet import DPRNN

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim = 256):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):  # x: (batch, seq_length, input_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        score = torch.bmm(q, k.transpose(1,2))/(self.input_dim**0.5)
        attention = self.softmax(score)
        weighted = torch.matmul(attention, v)
        return weighted
        

class FiLM(nn.Module):
    def __init__(self, film_meta, condition_size):
        super(FiLM, self).__init__()

        self.condition_size = condition_size

        self.modules, _ = self.create_film_modules(
            film_meta=film_meta, 
            ancestor_names=[],
        )
        
    def create_film_modules(self, film_meta, ancestor_names):

        modules = {}
       
        # Pre-order traversal of modules
        for module_name, value in film_meta.items():

            if isinstance(value, int):

                ancestor_names.append(module_name)
                unique_module_name = '->'.join(ancestor_names)

                modules[module_name] = self.add_film_layer_to_module(
                    num_features=value, 
                    unique_module_name=unique_module_name,
                )

            elif isinstance(value, dict):

                ancestor_names.append(module_name)
                
                modules[module_name], _ = self.create_film_modules(
                    film_meta=value, 
                    ancestor_names=ancestor_names,
                )

            ancestor_names.pop()

        return modules, ancestor_names

    def add_film_layer_to_module(self, num_features, unique_module_name):

        layer = nn.Linear(self.condition_size, num_features)
        init_layer(layer)
        self.add_module(name=unique_module_name, module=layer)

        return layer

    def forward(self, conditions):
        
        film_dict = self.calculate_film_data(
            conditions=conditions, 
            modules=self.modules,
        )

        return film_dict

    def calculate_film_data(self, conditions, modules):

        film_data = {}

        # Pre-order traversal of modules
        for module_name, module in modules.items():

            if isinstance(module, nn.Module):
                film_data[module_name] = module(conditions)[:, :, None, None]

            elif isinstance(module, dict):
                film_data[module_name] = self.calculate_film_data(conditions, module)

        return film_data


class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        momentum: float,
        has_film,
    ):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )
        
        # self.atten1 = SelfAttention(1, in_channels)   # added by HanYin
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )
        
        # self.atten2 = SelfAttention(1, out_channels)  # added by HanYin
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.has_film = has_film

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        b1 = film_dict['beta1']
        b2 = film_dict['beta2']
        
        # # === hanyin ===
        # b1_ = b1.reshape(b1.shape[0], b1.shape[1], 1)
        # b2_ = b2.reshape(b2.shape[0], b2.shape[1], 1)
        
        # b1_ = self.atten1(b1_).unsqueeze(-1)
        # b2_ = self.atten2(b2_).unsqueeze(-1)
        
        # b1 = b1 + b1_
        # b2 = b2 + b2_
        # # =============
        
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1, negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x) + b2, negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(input_tensor) + x
        else:
            return input_tensor + x


class EncoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        downsample: Tuple,
        momentum: float,
        has_film,
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes1B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, momentum, has_film,
        )
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        encoder = self.conv_block1(input_tensor, film_dict['conv_block1'])
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        upsample: Tuple,
        momentum: float,
        has_film,
    ):
        r"""Decoder block, contains 1 transposed convolutional and 8 convolutional layers."""
        super(DecoderBlockRes1B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, kernel_size, momentum, has_film,
        )
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.has_film = has_film

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor, film_dict: Dict,
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        # b1 = film_dict['beta1']

        b1 = film_dict['beta1']
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1))
        # (batch_size, input_feature_maps, time_steps, freq_bins)

        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

        x = self.conv_block2(x, film_dict['conv_block2'])
        # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

        return x


class ResUNet30_Base(nn.Module, Base):
    def __init__(self, input_channels, output_channels, dprnn=False, dprnn_layers=2, dprnn_hidden=128):
        super(ResUNet30_Base, self).__init__()

        window_size = 1024
        hop_size = 160
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.dprnn = dprnn
        self.dprnn_layers = dprnn_layers
        self.dprnn_hidden = dprnn_hidden

        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True,
        )

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()
        # --- hanyin ---
        if self.dprnn:
            self.DPRNN = nn.Sequential(DPRNN('GRU', 384, self.dprnn_hidden, 384, dropout = 0.1,
                                             num_layers=self.dprnn_layers))
        # --- han yin ---
    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform


    def forward(self, mixtures, film_dict):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet
        # Encoder
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x, film_dict['encoder_block1'])  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool, film_dict['encoder_block2'])  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool, film_dict['encoder_block3'])  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict['encoder_block4'])  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict['encoder_block5'])  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict['encoder_block6'])  # x6_pool: (bs, 384, T / 32, F / 64)
        x_center, _ = self.conv_block7a(x6_pool, film_dict['conv_block7a'])  # (bs, 384, T / 32, F / 64)
        
        # DPRNN Block
        if self.dprnn:
            x_center = self.DPRNN(x_center)
        # # # 
        
        # Decoder
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'])  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'])  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'])  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'])  # (bs, 32, T, F)

        x = self.after_conv(x12)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        output_dict = {'waveform': separated_audio}

        return output_dict


def get_film_meta(module):

    film_meta = {}

    if hasattr(module, 'has_film'):\

        if module.has_film:
            film_meta['beta1'] = module.bn1.num_features
            film_meta['beta2'] = module.bn2.num_features
        else:
            film_meta['beta1'] = 0
            film_meta['beta2'] = 0

    for child_name, child_module in module.named_children():

        child_meta = get_film_meta(child_module)

        if len(child_meta) > 0:
            film_meta[child_name] = child_meta
    
    return film_meta


class ResUNet30(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size,
                 dprnn=False, dprnn_layers=2, dprnn_hidden=128):
        super(ResUNet30, self).__init__()

        self.base = ResUNet30_Base(
            input_channels=input_channels, 
            output_channels=output_channels,
            dprnn=dprnn, 
            dprnn_layers=dprnn_layers, 
            dprnn_hidden=dprnn_hidden
        )
        
        self.film_meta = get_film_meta(
            module=self.base,
        )
        
        self.film = FiLM(
            film_meta=self.film_meta, 
            condition_size=condition_size
        )


    def forward(self, input_dict):
        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )
        
        output_dict = self.base(
            mixtures=mixtures, 
            film_dict=film_dict,
        )

        return output_dict

    @torch.no_grad()
    def chunk_inference(self, input_dict):
        chunk_config = {
                    'NL': 1.0,
                    'NC': 3.0,
                    'NR': 1.0,
                    'RATE': 32000
                }

        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])

        L = mixtures.shape[2]
        
        out_np = np.zeros([1, L])

        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]

            chunk_out = self.base(
                mixtures=chunk_in, 
                film_dict=film_dict,
            )['waveform']
            
            chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

            if current_idx == 0:
                out_np[:, current_idx:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, :-NR] if NR != 0 else chunk_out_np
            else:
                out_np[:, current_idx+NL:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, NL:-NR] if NR != 0 else chunk_out_np[:, NL:]

            current_idx += NC

            if current_idx < L:
                chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
                chunk_out = self.base(
                    mixtures=chunk_in, 
                    film_dict=film_dict,
                )['waveform']

                chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

                seg_len = chunk_out_np.shape[1]
                out_np[:, current_idx + NL:current_idx + seg_len] = \
                    chunk_out_np[:, NL:]

        return out_np

## ------------------ hanyin --------------------- ##
class ConvBlockRes_1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        momentum: float,
        has_film,
    ):
        r"""Residual block."""
        super(ConvBlockRes_1d, self).__init__()

        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.has_film = has_film

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        b1 = film_dict['beta1']
        b1 = b1.squeeze(-1)
        b2 = film_dict['beta2']
        b2 = b2.squeeze(-1)
        
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1, negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x) + b2, negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(input_tensor) + x
        else:
            return input_tensor + x

class EncoderBlockRes1B_1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        downsample: tuple,
        momentum: float,
        has_film,
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes1B_1d, self).__init__()

        self.conv_block1 = ConvBlockRes_1d(
            in_channels, out_channels, kernel_size, momentum, has_film,
        )
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        encoder = self.conv_block1(input_tensor, film_dict['conv_block1'])
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder

class DecoderBlockRes1B_1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample: int,
        momentum: float,
        has_film,
    ):
        r"""Decoder block, contains 1 transposed convolutional and 8 convolutional layers."""
        super(DecoderBlockRes1B_1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.bn1 = nn.BatchNorm1d(in_channels, momentum=momentum)
        self.conv_block2 = ConvBlockRes_1d(
            out_channels * 2, out_channels, kernel_size, momentum, has_film,
        )
        self.bn2 = nn.BatchNorm1d(in_channels, momentum=momentum)
        self.has_film = has_film

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor, film_dict: Dict,
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        # b1 = film_dict['beta1']

        b1 = film_dict['beta1']
        b1 = b1.squeeze(-1)
        
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1))
        # (batch_size, input_feature_maps, time_steps, freq_bins)
        # print(x.shape, concat_tensor.shape)
        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

        x = self.conv_block2(x, film_dict['conv_block2'])
        # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

        return x

class CrossAttention_1(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(CrossAttention_1, self).__init__()
        self.attention1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x_center, y_center, x_center1, y_center1):
        x_out, _ = self.attention1(x_center1, y_center1, y_center)
        x_out = x_out + x_center
        y_out, _ = self.attention2(y_center1, x_center1, x_center)
        y_out = y_out + y_center
        return x_out, y_out


class HResUNet30_Base_1(nn.Module, Base):
    def __init__(self, input_channels, output_channels):
        super(HResUNet30_Base_1, self).__init__()

        window_size = 1024
        hop_size = 160
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3
        self.alpha = nn.parameter.Parameter(torch.ones((2,)), requires_grad=True)
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True,
        )

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        
        # ----- waveform branch ----- #
        self.pre_conv_1d = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True,
        )
        
        self.encoder_block1_1d = EncoderBlockRes1B_1d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            downsample=(1,2),
            momentum=momentum,
            has_film=True,
        )
        
        self.encoder_block2_1d = EncoderBlockRes1B_1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            downsample=(1,2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block3_1d = EncoderBlockRes1B_1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            downsample=(1,4),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block4_1d = EncoderBlockRes1B_1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            downsample=(1,4),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block5_1d = EncoderBlockRes1B_1d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            downsample=(1,2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block6_1d = EncoderBlockRes1B_1d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            downsample=(1,2),
            momentum=momentum,
            has_film=True,
        )
        
        self.conv_block7a_1d = EncoderBlockRes1B_1d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
        )
        
        # decoders
        self.decoder_block1_1d = DecoderBlockRes1B_1d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            upsample=2,
            momentum=momentum,
            has_film=True,
        )
        
        self.decoder_block2_1d = DecoderBlockRes1B_1d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            upsample=2,
            momentum=momentum,
            has_film=True,
        )
        
        self.decoder_block3_1d = DecoderBlockRes1B_1d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            upsample=4,
            momentum=momentum,
            has_film=True,
        )
        
        self.decoder_block4_1d = DecoderBlockRes1B_1d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            upsample=4,
            momentum=momentum,
            has_film=True,
        )
        
        self.decoder_block5_1d = DecoderBlockRes1B_1d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            upsample=2,
            momentum=momentum,
            has_film=True,
        )
        
        self.decoder_block6_1d = DecoderBlockRes1B_1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            upsample=2,
            momentum=momentum,
            has_film=True,
        )
        
        self.after_conv_1d = nn.Conv1d(
            in_channels=32,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        # --- cross domain RNN ---
        self.spec_translayer1 = nn.TransformerEncoderLayer(d_model=256, nhead=8,
                                                          dim_feedforward=1024,
                                                          batch_first=True)
        self.spec_translayer2 = nn.TransformerEncoderLayer(d_model=256, nhead=8,
                                                          dim_feedforward=1024,
                                                          batch_first=True)
        self.spec_rnn1_ = nn.TransformerEncoder(self.spec_translayer1, num_layers=2)
        self.spec_rnn2_ = nn.TransformerEncoder(self.spec_translayer1, num_layers=2)
        
        # nn.GRU(input_size=256, hidden_size=128, num_layers=2, 
        #                         bidirectional=True, batch_first=True)
        
        self.wave_adapool_in = nn.Linear(625, 256) # nn.AdaptiveAvgPool1d(256)
        self.wave_adapool_out = nn.Linear(256, 625) # nn.AdaptiveAvgPool1d(625)
        
        self.wave_translayer1 = nn.TransformerEncoderLayer(d_model=256, nhead=8,
                                                          dim_feedforward=1024,
                                                          batch_first=True)
        self.wave_translayer2 = nn.TransformerEncoderLayer(d_model=256, nhead=8,
                                                          dim_feedforward=1024,
                                                          batch_first=True)
        self.wave_rnn1_ = nn.TransformerEncoder(self.spec_translayer1, num_layers=4)
        self.wave_rnn2_ = nn.TransformerEncoder(self.spec_translayer1, num_layers=4)
        
        self.cross_atten = CrossAttention_1(embed_dim=256, num_heads=8)
        
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)
        init_layer(self.pre_conv_1d)
        
    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform


    def forward(self, mixtures, film_dict):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        # === spectrogram encoder branch ===
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x, film_dict['encoder_block1'])  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool, film_dict['encoder_block2'])  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool, film_dict['encoder_block3'])  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict['encoder_block4'])  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict['encoder_block5'])  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict['encoder_block6'])  # x6_pool: (bs, 384, T / 32, F / 64)
        # (bs, 384, T / 32, F / 64) = (bs, 384, 32, 8)
        x_center, _ = self.conv_block7a(x6_pool, film_dict['conv_block7a'])  
        
        # === waveform encoder branch ===
        # norm
        wave = (mixtures - torch.mean(mixtures))/(torch.var(mixtures))
        wave = self.pre_conv_1d(wave)  # [bs, 32, Frames]
        y1_pool, y1 = self.encoder_block1_1d(wave, film_dict['encoder_block1']) # y1_pool:[bs, 32, Frames/2] y1:[bs, 32, Frames]
        y2_pool, y2 = self.encoder_block2_1d(y1_pool, film_dict['encoder_block2']) # y2_pool:[bs, 64, Frames/4] y2:[bs, 64, Frames/2]
        y3_pool, y3 = self.encoder_block3_1d(y2_pool, film_dict['encoder_block3']) # y3_pool:[bs, 128, Frames/16] y3:[bs, 128, Frames/4]
        y4_pool, y4 = self.encoder_block4_1d(y3_pool, film_dict['encoder_block4']) # y4_pool:[bs, 256, Frames/64] y4:[bs, 256, Frames/16]
        y5_pool, y5 = self.encoder_block5_1d(y4_pool, film_dict['encoder_block5']) # y5_pool:[bs, 384, Frames/128] y5:[bs, 384, Frames/64]
        y6_pool, y6 = self.encoder_block6_1d(y5_pool, film_dict['encoder_block6']) # y6_pool:[bs, 384, Frames/256] y6:[bs, 384, Frames/128]
        # y_center:[bs, 384, Frames/64] = [bs, 384, 312]
        y_center, _ = self.conv_block7a_1d(y6_pool, film_dict['conv_block7a'])  # (bs, 384, T / 32, F / 64)
        
        # === cross domain RNN ===
        B_, C_, T_, F_ = x_center.shape[0], x_center.shape[1], \
                     x_center.shape[2], x_center.shape[3]
        x_center = x_center.reshape((B_, C_, -1)) # [bs, 384, 256]
        y_center = self.wave_adapool_in(y_center) # [bs, 384, 256]
        
        x_center1 = self.spec_rnn1_(x_center)
        y_center1 = self.wave_rnn1_(y_center)
        
        x_center, y_center = self.cross_atten(x_center, y_center, x_center1, y_center1)
        
        x_center = self.spec_rnn2_(x_center)
        y_center = self.wave_rnn2_(y_center)
        
        x_center = x_center.reshape((B_, C_, T_, F_))
        y_center = self.wave_adapool_out(y_center)
        
        # === spectrogram decoder branch ===
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'])  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'])  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'])  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'])  # (bs, 32, T, F)
        
        x = self.after_conv(x12)
        
        # === waveform decoder branch ===
        y7 = self.decoder_block1_1d(y_center, y6, film_dict['decoder_block1'])
        y8 = self.decoder_block2_1d(y7, y5, film_dict['decoder_block2'])
        y9 = self.decoder_block3_1d(y8, y4, film_dict['decoder_block3'])
        y10 = self.decoder_block4_1d(y9, y3, film_dict['decoder_block4'])
        y11 = self.decoder_block5_1d(y10, y2, film_dict['decoder_block5'])
        y12 = self.decoder_block6_1d(y11, y1, film_dict['decoder_block6'])
        
        y = self.after_conv_1d(y12)
        
        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)
        
        separated_audio_final = self.alpha[0] * separated_audio + self.alpha[1] * y
        
        output_dict = {'waveform': separated_audio_final,
                       'spec_out': separated_audio,
                       'wave_out': y}

        return output_dict


class HResUNet30_Base(nn.Module, Base):
    def __init__(self, input_channels, output_channels):
        super(HResUNet30_Base, self).__init__()

        window_size = 1024
        hop_size = 160
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels
        self.target_sources_num = 1
        self.K = 3
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True,
        )

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        
        # ----- waveform branch ----- #
        self.pre_conv_wave = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=32, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True,
        )
        
        self.encoder_block1_wave = EncoderBlockRes1B_1d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            downsample=(1,2),
            momentum=momentum,
            has_film=True,
        )
        
        self.encoder_block2_wave = EncoderBlockRes1B_1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            downsample=(1,2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block3_wave = EncoderBlockRes1B_1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            downsample=(1,4),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block4_wave = EncoderBlockRes1B_1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            downsample=(1,4),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block5_wave = EncoderBlockRes1B_1d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            downsample=(1,2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block6_wave = EncoderBlockRes1B_1d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            downsample=(1,2),
            momentum=momentum,
            has_film=True,
        )
        
        self.conv_block7a_wave = EncoderBlockRes1B_1d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
        )
        
        
        # --- cross domain RNN ---
        self.cross_spec_fc = nn.Linear(in_features=256, out_features=256,
                                       bias=True)
        
        self.cross_wave_fc = nn.Linear(in_features=625, out_features=256,
                                       bias=True)
        
        self.cross_BiGRU_ = nn.GRU(input_size=256, hidden_size=128, 
                                  bidirectional=True, batch_first=True,
                                  num_layers=2)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, 
                                               batch_first=True)
        
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)
        init_layer(self.pre_conv_wave)
        
    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform


    def forward(self, mixtures, film_dict):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        # === spectrogram encoder branch ===
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x, film_dict['encoder_block1'])  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool, film_dict['encoder_block2'])  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool, film_dict['encoder_block3'])  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict['encoder_block4'])  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict['encoder_block5'])  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict['encoder_block6'])  # x6_pool: (bs, 384, T / 32, F / 64)
        # (bs, 384, T / 32, F / 64) = (bs, 384, 32, 8)
        x_center, _ = self.conv_block7a(x6_pool, film_dict['conv_block7a'])  
        
        # === waveform encoder branch ===
        # norm
        wave = (mixtures - torch.mean(mixtures))/(torch.var(mixtures))
        wave = self.pre_conv_wave(wave)  # [bs, 32, Frames]
        y1_pool, _ = self.encoder_block1_wave(wave, film_dict['encoder_block1']) # y1_pool:[bs, 32, Frames/2] y1:[bs, 32, Frames]
        y2_pool, _ = self.encoder_block2_wave(y1_pool, film_dict['encoder_block2']) # y2_pool:[bs, 64, Frames/4] y2:[bs, 64, Frames/2]
        y3_pool, _ = self.encoder_block3_wave(y2_pool, film_dict['encoder_block3']) # y3_pool:[bs, 128, Frames/16] y3:[bs, 128, Frames/4]
        y4_pool, _ = self.encoder_block4_wave(y3_pool, film_dict['encoder_block4']) # y4_pool:[bs, 256, Frames/64] y4:[bs, 256, Frames/16]
        y5_pool, _ = self.encoder_block5_wave(y4_pool, film_dict['encoder_block5']) # y5_pool:[bs, 384, Frames/128] y5:[bs, 384, Frames/64]
        y6_pool, _ = self.encoder_block6_wave(y5_pool, film_dict['encoder_block6']) # y6_pool:[bs, 384, Frames/256] y6:[bs, 384, Frames/128]
        # y_center:[bs, 384, Frames/64] = [bs, 384, 312]
        y_center, _ = self.conv_block7a_wave(y6_pool, film_dict['conv_block7a'])  # (bs, 384, T / 32, F / 64)
        
        # === cross domain RNN ===
        B_, C_, T_, F_ = x_center.shape[0], x_center.shape[1], \
                     x_center.shape[2], x_center.shape[3]
        x_center = x_center.reshape((B_, C_, -1)) # [bs, 384, 256]
        x_center1 = self.cross_spec_fc(x_center)
        
        y_center1 = self.cross_wave_fc(y_center) # [bs, 384, 256]        
        
        x_center2, _ = self.cross_attention(y_center1, x_center1, x_center)
        
        x_center, _ = self.cross_BiGRU_(x_center2)
        
        x_center = x_center.reshape((B_, C_, T_, F_))
        
        # === spectrogram decoder branch ===
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'])  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'])  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'])  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'])  # (bs, 32, T, F)
        
        x = self.after_conv(x12)
        
        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)
        
        output_dict = {'waveform': separated_audio}

        return output_dict
    

class HResUNet30(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size):
        super(HResUNet30, self).__init__()

        self.base = HResUNet30_Base(
            input_channels=input_channels, 
            output_channels=output_channels,
        )
        
        self.film_meta = get_film_meta(
            module=self.base,
        )
        
        self.film = FiLM(
            film_meta=self.film_meta, 
            condition_size=condition_size
        )


    def forward(self, input_dict):
        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )
        
        output_dict = self.base(
            mixtures=mixtures, 
            film_dict=film_dict,
        )

        return output_dict

    @torch.no_grad()
    def chunk_inference(self, input_dict):
        chunk_config = {
                    'NL': 1.0,
                    'NC': 3.0,
                    'NR': 1.0,
                    'RATE': 32000
                }

        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])

        L = mixtures.shape[2]
        
        out_np = np.zeros([1, L])

        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]

            chunk_out = self.base(
                mixtures=chunk_in, 
                film_dict=film_dict,
            )['waveform']
            
            chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

            if current_idx == 0:
                out_np[:, current_idx:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, :-NR] if NR != 0 else chunk_out_np
            else:
                out_np[:, current_idx+NL:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, NL:-NR] if NR != 0 else chunk_out_np[:, NL:]

            current_idx += NC

            if current_idx < L:
                chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
                chunk_out = self.base(
                    mixtures=chunk_in, 
                    film_dict=film_dict,
                )['waveform']

                chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

                seg_len = chunk_out_np.shape[1]
                out_np[:, current_idx + NL:current_idx + seg_len] = \
                    chunk_out_np[:, NL:]

        return out_np


if __name__ == '__main__':
    # checkpoint_path = '/home/bfzystudent/Personal/YH/DCASE/2024/Task9:LASS/codes/workspace/checkpoints/baseline_fintune_FSD50K_Clotho_Audiocaps_valid/model-epoch=190-val_loss=0.0150.ckpt'
    # checkpoint_data = torch.load(checkpoint_path,map_location='cpu')
    # model_params = checkpoint_data['state_dict']
    
    # resnet = HResUNet30(input_channels=1, output_channels=1, condition_size=512)
    # resnet.eval()
    # resnet2 = ResUNet30(input_channels=1, output_channels=1, condition_size=512)
    # resnet2.eval()
    
    # audio = torch.rand((6, 1, 160000))
    # caption = torch.ones((6, 512))
    # input_dict = {'mixture': audio,
    #               'condition': caption}
    # with torch.no_grad():
    #     # output = resnet(input_dict)
    #     # sep_audio = output['waveform']
    #     output2 = resnet2(input_dict)
    #     sep_audio2 = output2['waveform']
        
    #     # print(sep_audio.shape)
    #     print(sep_audio2.shape)
    model = ResUNet30(input_channels=1, output_channels=1, condition_size=512, 
                        dprnn=True, dprnn_layers=6, dprnn_hidden=256)
    import numpy as np
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('\nTotal paramters: ' + str(model_params))
    
    
    
    
    
    
