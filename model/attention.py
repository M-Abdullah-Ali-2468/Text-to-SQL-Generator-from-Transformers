import torch
import torch.nn as nn


class Scaled_dot_product_Attention(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,input,wq,wk,wv,dk,input_kv,masking,padding_mask):

    q=wq(input)

    if input_kv is not None:
      input=input_kv

    k=wk(input)
    v=wv(input)

    qk_t=torch.matmul(q,k.transpose(-2,-1))
    qk_t=qk_t/torch.sqrt(torch.tensor(dk,dtype=torch.float32,device=q.device))

    if padding_mask is not None:
      qk_t=qk_t.masked_fill(padding_mask,float('-inf'))

    if masking==True:
      masked=torch.triu(torch.ones(qk_t.shape[-2:],device=input.device),diagonal=1)
      masked=masked.masked_fill(masked==1,float('-inf'))
      qk_t=qk_t+masked

    attention_score=torch.softmax(qk_t,dim=-1)

    attention_output=torch.matmul(attention_score,v)

    return attention_output,attention_score


class Multi_Head_Attention(nn.Module):
  def __init__(self,heads,dmodel):
    super().__init__()

    self.dmodel=dmodel
    self.heads=heads

    self.dh=self.dmodel//self.heads
    self.dk=self.dh
    self.dv=self.dh

    self.WQ=nn.ModuleList()
    self.WK=nn.ModuleList()
    self.WV=nn.ModuleList()

    self.WO=nn.Linear(self.dmodel,self.dmodel)

    for i in range(heads):
      self.WQ.append(nn.Linear(self.dmodel,self.dk))
      self.WK.append(nn.Linear(self.dmodel,self.dk))
      self.WV.append(nn.Linear(self.dmodel,self.dv))

    self.Scaled_dot_product_Attention=Scaled_dot_product_Attention()

  def forward(self,input,input_kv=None,masking=False,padding_mask=None):

    self.attention_output=[]
    self.attention_weights=[]

    for i in range(len(self.WQ)):
      attention_output,attention_weights=self.Scaled_dot_product_Attention(input,self.WQ[i],self.WK[i],self.WV[i],self.dk,input_kv,masking,padding_mask)

      self.attention_output.append(attention_output)
      self.attention_weights.append(attention_weights)

    attention_output=torch.cat(self.attention_output,dim=2)

    output=self.WO(attention_output)

    return output,self.attention_weights