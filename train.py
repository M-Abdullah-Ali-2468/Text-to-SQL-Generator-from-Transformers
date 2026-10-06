import os
import csv
import torch
import torch.nn as nn
import sentencepiece as spm

from starter.dataset import make_loader
from model.transformer import Transformers


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp=spm.SentencePieceProcessor(model_file="sql_sp.model")

train_loader=make_loader("train_pairs.jsonl",sp,train=True)
dev_loader=make_loader("dev_pairs.jsonl",sp,train=False)

transformer=Transformers(enc_layers=3,dec_layers=3,dmodel=256,heads=4,dff=1024,dropout=0.1).to(device)

optimizer=torch.optim.Adam(transformer.parameters(),betas=(0.9,0.98),eps=1e-9)

loss_function=nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)


def get_lr(step,dmodel=256,warmup_steps=4000):

  return dmodel**(-0.5)*min(step**(-0.5),step*warmup_steps**(-1.5))


previous_validation_loss=torch.inf

training_logs=[]

os.makedirs("checkpoints",exist_ok=True)

global_step=0


for epoch in range(20):

  losses=[]

  transformer.train()

  print("Epoch:",epoch+1,"/",20)

  for batch,(src,tgt) in enumerate(train_loader):

    src=src.to(device)
    tgt=tgt.to(device)

    global_step+=1

    lr=get_lr(global_step)

    for param_group in optimizer.param_groups:
      param_group["lr"]=lr

    optimizer.zero_grad()

    decoder_input=tgt[:,:-1]
    target=tgt[:,1:]

    logits=transformer(src,decoder_input)

    loss=loss_function(logits.reshape(-1,logits.size(-1)),target.reshape(-1))

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

  average_training_loss=sum(losses)/len(losses)

  print("Average Training Loss:",average_training_loss)

  transformer.eval()

  validation_loss=[]

  with torch.no_grad():

    for batch,(src,tgt) in enumerate(dev_loader):

      src=src.to(device)
      tgt=tgt.to(device)

      decoder_input=tgt[:,:-1]
      target=tgt[:,1:]

      logits=transformer(src,decoder_input)

      loss=loss_function(logits.reshape(-1,logits.size(-1)),target.reshape(-1))

      validation_loss.append(loss.item())

  average_validation_loss=sum(validation_loss)/len(validation_loss)

  print("Average Validation Loss:",average_validation_loss)
  print("Learning Rate:",lr)

  if previous_validation_loss>average_validation_loss:

    previous_validation_loss=average_validation_loss

    print("checkpoint saved!")

    training_logs.append([epoch+1,average_training_loss,average_validation_loss,lr,"Yes"])

    torch.save({"epoch":epoch+1,"model_state_dict":transformer.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"validation_loss":average_validation_loss,"learning_rate":lr,"global_step":global_step},"checkpoints/best_transformer.pt")

  else:

    training_logs.append([epoch+1,average_training_loss,average_validation_loss,lr,"No"])


with open("checkpoints/training_logs.csv","w",newline="") as file:

  writer=csv.writer(file)

  writer.writerow(["Epoch","Training Loss","Validation Loss","Learning Rate","Checkpoint Saved"])

  writer.writerows(training_logs)