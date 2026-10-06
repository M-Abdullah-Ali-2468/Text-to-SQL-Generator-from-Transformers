import torch
import torch.nn as nn
import sentencepiece as spm

from starter.embeddings import InputLayer,TokenEmbedding
from model.layers import Encoder,Decoder


sp=spm.SentencePieceProcessor(model_file="sql_sp.model")


class Transformers(nn.Module):
  def __init__(self,enc_layers,dec_layers,dmodel,heads,dff,dropout=0.1):
    super().__init__()

    self.enc_layers=enc_layers
    self.dec_layers=dec_layers
    self.dmodel=dmodel
    self.heads=heads
    self.dff=dff
    self.dropout=dropout

    self.shared=TokenEmbedding(sp.get_piece_size(),self.dmodel)

    self.input_layer=InputLayer(self.shared,self.dmodel)

    self.encoder=Encoder(self.enc_layers,self.dmodel,self.heads,self.dff,False,self.dropout)

    self.decoder=Decoder(self.dec_layers,self.dmodel,self.heads,self.dff,self.dropout)

    self.output_projection=nn.Linear(self.dmodel,sp.get_piece_size(),bias=False)

    self.output_projection.weight=self.shared.emb.weight

  def forward(self,src,tgt):

    src_padding_mask=(src==0).unsqueeze(1)
    tgt_padding_mask=(tgt==0).unsqueeze(1)

    batch_src=self.input_layer(src)
    batch_tgt=self.input_layer(tgt)

    encoder_output=self.encoder(batch_src,src_padding_mask)

    decoder_output=self.decoder(batch_tgt,encoder_output,tgt_padding_mask,src_padding_mask)

    logits=self.output_projection(decoder_output)

    return logits