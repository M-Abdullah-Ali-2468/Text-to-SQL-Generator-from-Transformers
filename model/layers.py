import torch
import torch.nn as nn

from model.attention import Multi_Head_Attention


class Point_wise_feed_forward(nn.Module):
  def __init__(self,dmodel,dff):
    super().__init__()

    self.dmodel=dmodel
    self.dff=dff

    self.linear1=nn.Linear(self.dmodel,self.dff)
    self.linear2=nn.Linear(self.dff,self.dmodel)

  def forward(self,input):

    feed1=torch.relu(self.linear1(input))
    feed2=self.linear2(feed1)

    return feed2


class Encoder_layer(nn.Module):
  def __init__(self,dmodel,heads,dff,dropout):
    super().__init__()

    self.dmodel=dmodel
    self.heads=heads
    self.dff=dff
    self.dropout=nn.Dropout(dropout)

    self.Multi_Head_Attention=Multi_Head_Attention(self.heads,self.dmodel)

    self.Point_wise_feed_forward=Point_wise_feed_forward(self.dmodel,self.dff)

    self.layer_norm_1=nn.LayerNorm(self.dmodel)
    self.layer_norm_2=nn.LayerNorm(self.dmodel)

  def forward(self,input,padding_mask):

    multi_head_attention_output,attention_weights=self.Multi_Head_Attention(input,padding_mask=padding_mask)

    multi_head_attention_output=self.dropout(multi_head_attention_output)

    layer_norm_1=self.layer_norm_1(input+multi_head_attention_output)

    pwff_ouptut=self.Point_wise_feed_forward(layer_norm_1)

    pwff_ouptut=self.dropout(pwff_ouptut)

    layer_norm_2=self.layer_norm_2(layer_norm_1+pwff_ouptut)

    return layer_norm_2


class Encoder(nn.Module):
  def __init__(self,n_layers,dmodel,heads,dff,masking,dropout):
    super().__init__()

    self.n_layers=n_layers
    self.dmodel=dmodel
    self.heads=heads
    self.dff=dff
    self.masking=masking
    self.dropout=dropout

    self.layers=nn.ModuleList()

    for i in range(n_layers):
      self.layers.append(Encoder_layer(self.dmodel,self.heads,self.dff,self.dropout))

  def forward(self,batch,padding_mask):

    for layer in self.layers:
      batch=layer(batch,padding_mask)

    return batch


class Decoder_layer(nn.Module):
  def __init__(self,dmodel,heads,dff,dropout):
    super().__init__()

    self.dmodel=dmodel
    self.heads=heads
    self.dff=dff
    self.dropout=nn.Dropout(dropout)

    self.Multi_Head_Attention_1=Multi_Head_Attention(self.heads,self.dmodel)
    self.Multi_Head_Attention_2=Multi_Head_Attention(self.heads,self.dmodel)

    self.Point_wise_feed_forward=Point_wise_feed_forward(self.dmodel,self.dff)

    self.layer_norm_1=nn.LayerNorm(self.dmodel)
    self.layer_norm_2=nn.LayerNorm(self.dmodel)
    self.layer_norm_3=nn.LayerNorm(self.dmodel)

  def forward(self,input,encoder_output,tgt_padding_mask,src_padding_mask):

    masked_attention_output,masked_attention_weights=self.Multi_Head_Attention_1(input,masking=True,padding_mask=tgt_padding_mask)

    masked_attention_output=self.dropout(masked_attention_output)

    layer_norm_1=self.layer_norm_1(input+masked_attention_output)

    cross_attention_output,cross_attention_weights=self.Multi_Head_Attention_2(layer_norm_1,encoder_output,padding_mask=src_padding_mask)

    cross_attention_output=self.dropout(cross_attention_output)

    layer_norm_2=self.layer_norm_2(layer_norm_1+cross_attention_output)

    pwff_ouptut=self.Point_wise_feed_forward(layer_norm_2)

    pwff_ouptut=self.dropout(pwff_ouptut)

    layer_norm_3=self.layer_norm_3(layer_norm_2+pwff_ouptut)

    return layer_norm_3


class Decoder(nn.Module):
  def __init__(self,n_layers,dmodel,heads,dff,dropout):
    super().__init__()

    self.n_layers=n_layers
    self.dmodel=dmodel
    self.heads=heads
    self.dff=dff
    self.dropout=dropout

    self.layers=nn.ModuleList()

    for i in range(n_layers):
      self.layers.append(Decoder_layer(self.dmodel,self.heads,self.dff,self.dropout))

  def forward(self,input,encoder_output,tgt_padding_mask,src_padding_mask):

    x=input

    for layer in self.layers:
      x=layer(x,encoder_output,tgt_padding_mask,src_padding_mask)

    return x