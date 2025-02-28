from layers.TDformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, AttentionLayer, series_decomp, series_decomp_multi,series_decomp_res,series_decomp_multi_res
import torch.nn as nn
import torch
from layers.Embed import DataEmbedding
from layers.Attention import WaveletAttention, FourierAttention, FullAttention
from layers.RevIN import RevIN
import torch.nn.functional as F

class GatedLayer(nn.Module):
    def __init__(self, d_model):
        super(GatedLayer, self).__init__()
        self.gate = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.gate(x))
        return x * gate
    
class Model(nn.Module):
    """
    Transformer for seasonality, MLP for trend
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.output_stl = configs.output_stl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi_res(kernel_size)  
        else:
            self.decomp = series_decomp_res(kernel_size)

        # Embedding
        self.enc_seasonal_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.dec_seasonal_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.enc_trend_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.dec_trend_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder
        if configs.version == 'Wavelet':
            enc_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len,
                                                  seq_len_kv=configs.seq_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                  seq_len_kv=configs.seq_len // 2 + configs.pred_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = WaveletAttention(in_channels=configs.d_model,
                                                   out_channels=configs.d_model,
                                                   seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                   seq_len_kv=configs.seq_len,
                                                   ich=configs.d_model,
                                                   T=configs.temp,
                                                   activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Fourier':
            enc_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)#
            dec_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Time':
            enc_self_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_self_attention = FullAttention(True, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_cross_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention)

        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        enc_self_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.seasonal_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        dec_self_attention,
                        configs.d_model),
                    AttentionLayer(
                        dec_cross_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )


        
        # Trend
        enc_self_attention_trend=FullAttention(False, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
        
        dec_self_attention_trend=FullAttention(True, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
        
        dec_cross_attention_trend=FullAttention(False, T=configs.temp, activation=configs.activation,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention)
        
        
        self.trend_encoder = Encoder(
            [EncoderLayer(AttentionLayer(enc_self_attention_trend, configs.d_model), configs.d_model, dropout=configs.dropout, activation=configs.activation) for l in range(2)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.trend_decoder = Decoder(
            [DecoderLayer(AttentionLayer(dec_self_attention_trend, configs.d_model), AttentionLayer(dec_cross_attention_trend, configs.d_model), configs.d_model, dropout=configs.dropout, activation=configs.activation) for l in range(1)],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.revin_trend = RevIN(configs.enc_in).to(self.device)
        #
        self.residual_lstm = nn.LSTM(configs.enc_in, configs.d_model, batch_first=True)
        self.residual_proj = nn.Linear(configs.d_model, configs.c_out)
        
        
        #
        self.residual_mlp = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )
        self.residual_Linear=nn.Linear(configs.seq_len,configs.pred_len)
        
        self.full_attention = FullAttention(mask_flag=False, T=configs.temp, activation=configs.activation, output_attention=False)

        self.finalprojection = nn.Sequential(
            nn.Linear(configs.enc_in, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.projector2 = nn.Linear(configs.enc_in, 2, bias=True)
        
        
        # Gated Layers
        self.seasonal_gate = GatedLayer(configs.c_out)
        self.trend_gate = GatedLayer(configs.c_out)
        self.residual_gate = GatedLayer(configs.c_out)
        


        
        
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
 
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(self.device) 
    


        seasonal_enc, trend_enc, residual_enc = self.decomp(x_enc)
        


        # 防止 NaN 值，设置一个最小值
        seasonal_mean_abs = seasonal_enc.abs().mean(dim=1)  # 季节性成分的平均绝对值
        min_value = 1e-6
        seasonal_mean_abs = torch.where(seasonal_mean_abs == 0, torch.tensor(min_value).to(self.device), seasonal_mean_abs)
        seasonal_dec = F.pad(seasonal_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))    
        

        enc_out = self.enc_seasonal_embedding(seasonal_enc, x_mark_enc)  
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask) 
        
        dec_out = self.dec_seasonal_embedding(seasonal_dec, x_mark_dec)
        
        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        seasonal_out = seasonal_out[:, -self.pred_len:, :]
        

        seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)
        

        seasonal_out = self.seasonal_gate(seasonal_out)

        # trend
        trend_dec=F.pad(trend_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        

        
        trend_enc_out = self.enc_trend_embedding(trend_enc, x_mark_enc)
        trend_enc_out, _ = self.trend_encoder(trend_enc_out, attn_mask=None)
        trend_dec_out = self.dec_trend_embedding(trend_dec, x_mark_dec)
        trend_out, _ = self.trend_decoder(trend_dec_out, trend_enc_out, x_mask=None, cross_mask=None)
        trend_out = trend_out[:, -self.pred_len:, :]
        
        # Apply gate to trend output
        trend_out = self.trend_gate(trend_out)
        
        

        residual_out=self.revin_trend(residual_enc, 'norm')
        residual_out = self.residual_mlp(residual_enc.permute(0, 2, 1)).permute(0, 2, 1)
        residual_out=self.revin_trend(residual_out, 'norm')

        # LSTM
        residual_out, _ = self.residual_lstm(residual_out)
        residual_out = self.residual_proj(residual_out)
        
        if self.pred_len > seasonal_enc.shape[1]:
            residual_out = F.interpolate(residual_out.permute(0, 2, 1), size=self.pred_len, mode='linear', align_corners=False).permute(0, 2, 1)
        else:
            residual_out = residual_out[:, -self.pred_len:, :]
            
        # Apply gate to residual output
        residual_out = self.residual_gate(residual_out)

        residual=residual_out    
        
#         dec_out= trend_out  + seasonal_out
        
        dec_out= trend_out + residual_out + seasonal_out
        
        



        
        
        dec_out = dec_out.permute(1, 0, 2)
        B, L, D = dec_out.shape
        dec_out = dec_out.unsqueeze(2)
        dec_out, _ = self.full_attention(dec_out, dec_out, dec_out, attn_mask=None)
        
        dec_out = dec_out.squeeze(2) 
        
        dec_out = dec_out.permute(1, 0, 2) # Reshape back to original shape


        # ACC
        dec_out=self.projector2(dec_out)
        output=dec_out[:, -self.pred_len:, :]
        
        
        
        output = F.elu((output))
        output=F.log_softmax(output, dim=1)
        return output
