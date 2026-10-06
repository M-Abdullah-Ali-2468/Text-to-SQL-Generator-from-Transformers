import json
import os
import subprocess
import sys
import torch
import matplotlib.pyplot as plt
import sentencepiece as spm
from model.transformers import Transformers
from starter.dataset import make_loader
from decoding import greedy_decode,beam_search_decode,parse_sql_string,write_predictions
sp=spm.SentencePieceProcessor(model_file="sql_sp.model")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGG_OPS=["","MAX","MIN","COUNT","SUM","AVG"]
COND_OPS=["=",">","<"]
def load_model_and_data():
    dev_loader=make_loader("dev_pairs.jsonl",sp,False)
    test_loader=make_loader("test_pairs.jsonl",sp,False)
    transformer=Transformers(enc_layers=3,dec_layers=3,dmodel=256,heads=4,dff=1024,dropout=0.1).to(device)
    checkpoint=torch.load("checkpoints/best_transformer.pt",map_location=device)
    transformer.load_state_dict(checkpoint["model_state_dict"])
    transformer.eval()
    print("Model loaded successfully!")
    print("Epoch:",checkpoint["epoch"])
    print("Validation Loss:",checkpoint["validation_loss"])
    return transformer,dev_loader,test_loader
def dev_data_check(dev_loader):
    print("\nDEV DATA CHECK")
    print("Dev examples:",len(dev_loader.dataset))
    src,tgt=next(iter(dev_loader))
    print("src shape:",src.shape)
    print("tgt shape:",tgt.shape)
def prediction_files_check():
    print("\nCHECK PREDICTION FILES")
    print("Greedy:",os.path.exists("results/dev_greedy.jsonl"))
    print("Beam:",os.path.exists("results/dev_beam.jsonl"))
    print("Test Beam:",os.path.exists("results/test_beam.jsonl"))
    if os.path.exists("results/dev_greedy.jsonl"):
        print("Greedy lines:",sum(1 for _ in open("results/dev_greedy.jsonl",encoding="utf-8")))
    if os.path.exists("results/dev_beam.jsonl"):
        print("Beam lines:",sum(1 for _ in open("results/dev_beam.jsonl",encoding="utf-8")))
    if os.path.exists("results/test_beam.jsonl"):
        print("Test Beam lines:",sum(1 for _ in open("results/test_beam.jsonl",encoding="utf-8")))
def official_evaluator(gold_file,db_file,pred_file):
    subprocess.run([sys.executable,"WikiSQL/evaluate.py",gold_file,db_file,pred_file],check=False)
def official_dev_evaluation():
    print("\nOFFICIAL WIKISQL DEV EVALUATION")
    official_evaluator("WikiSQL/data/dev.jsonl","WikiSQL/data/dev.db","results/dev_greedy.jsonl")
    official_evaluator("WikiSQL/data/dev.jsonl","WikiSQL/data/dev.db","results/dev_beam.jsonl")
def parse_failure_rate():
    print("\nPARSE FAILURE RATE")
    for name in ["dev_greedy","dev_beam","test_beam"]:
        path=f"results/{name}.jsonl"
        total=0
        parse_errors=0
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                total+=1
                obj=json.loads(line)
                if obj.get("error")=="parse":
                    parse_errors+=1
        print(f"{name}:")
        print(f"Total: {total}")
        print(f"Parse failures: {parse_errors}")
        print(f"Parse failure rate: {parse_errors/total*100:.2f}%")
def normalize_conds(conds):
    return sorted([(c[0],c[1],str(c[2])) for c in conds])
def component_accuracy():
    print("\nCOMPONENT ACCURACY")
    GOLD_FILE="WikiSQL/data/dev.jsonl"
    PRED_FILES={"Greedy":"results/dev_greedy.jsonl","Beam":"results/dev_beam.jsonl"}
    gold_queries=[]
    with open(GOLD_FILE,"r",encoding="utf-8") as f:
        for line in f:
            gold_queries.append(json.loads(line)["sql"])
    for name,pred_file in PRED_FILES.items():
        sel_correct=0
        agg_correct=0
        where_correct=0
        total=len(gold_queries)
        with open(pred_file,"r",encoding="utf-8") as f:
            for gold,line in zip(gold_queries,f):
                pred_obj=json.loads(line)
                if "error" in pred_obj:
                    continue
                pred=pred_obj["query"]
                if pred["sel"]==gold["sel"]:
                    sel_correct+=1
                if pred["agg"]==gold["agg"]:
                    agg_correct+=1
                if normalize_conds(pred["conds"])==normalize_conds(gold["conds"]):
                    where_correct+=1
        print(f"\n{name}")
        print("-"*30)
        print(f"SELECT column accuracy: {sel_correct/total*100:.2f}%")
        print(f"Aggregation accuracy: {agg_correct/total*100:.2f}%")
        print(f"WHERE clause accuracy: {where_correct/total*100:.2f}%")
def causal_mask_check(transformer,dev_loader):
    print("\nCAUSAL MASK CHECK")
    transformer.eval()
    src,tgt=next(iter(dev_loader))
    src=src.to(device)
    tgt=tgt.to(device)
    dec_input=tgt[:,:-1].clone()
    src_padding_mask=(src==0).unsqueeze(1)
    with torch.no_grad():
        batch_src=transformer.input_layer(src)
        encoder_output=transformer.encoder(batch_src,src_padding_mask)
        batch_tgt=transformer.input_layer(dec_input)
        tgt_padding_mask=(dec_input==0).unsqueeze(1)
        output_1=transformer.decoder(batch_tgt,encoder_output,tgt_padding_mask,src_padding_mask)
    modified_input=dec_input.clone()
    modified_input[:,-1]=(modified_input[:,-1]+1)%sp.get_piece_size()
    with torch.no_grad():
        modified_batch_tgt=transformer.input_layer(modified_input)
        modified_tgt_padding_mask=(modified_input==0).unsqueeze(1)
        output_2=transformer.decoder(modified_batch_tgt,encoder_output,modified_tgt_padding_mask,src_padding_mask)
    difference=torch.abs(output_1[:,:-1]-output_2[:,:-1]).max().item()
    print("Maximum difference in earlier positions:",difference)
    if difference<1e-6:
        print("CAUSAL MASK CHECK: PASSED")
    else:
        print("CAUSAL MASK CHECK: FAILED")
def padding_mask_check(transformer,dev_loader):
    print("\nPADDING MASK CHECK")
    transformer.eval()
    src,tgt=next(iter(dev_loader))
    src=src.to(device)
    tgt=tgt.to(device)
    src_mask=(src==0).unsqueeze(1)
    with torch.no_grad():
        src_emb=transformer.input_layer(src)
        enc_out_1=transformer.encoder(src_emb,src_mask)
    extra_pads=torch.zeros((src.size(0),10),dtype=torch.long,device=device)
    src_padded=torch.cat([src,extra_pads],dim=1)
    src_mask_padded=(src_padded==0).unsqueeze(1)
    with torch.no_grad():
        src_emb_padded=transformer.input_layer(src_padded)
        enc_out_2=transformer.encoder(src_emb_padded,src_mask_padded)
    original_length=src.size(1)
    valid_positions=(src!=0).unsqueeze(-1)
    original_part=enc_out_1
    padded_part=enc_out_2[:,:original_length,:]
    difference=torch.abs(original_part-padded_part)
    valid_difference=difference.masked_select(valid_positions.expand_as(difference))
    max_difference=valid_difference.max().item()
    mean_difference=valid_difference.mean().item()
    print("Maximum difference:",max_difference)
    print("Mean difference:",mean_difference)
    if max_difference<1e-4:
        print("PADDING MASK CHECK: PASSED")
    else:
        print("PADDING MASK CHECK: FAILED")
def attention_row_sum_check(transformer,dev_loader):
    print("\nATTENTION ROW SUM CHECK")
    transformer.eval()
    src,tgt=next(iter(dev_loader))
    src=src.to(device)
    tgt=tgt.to(device)
    src_mask=(src==0).unsqueeze(1)
    with torch.no_grad():
        src_emb=transformer.input_layer(src)
        _,attention_weights=transformer.encoder.layers[0].Multi_Head_Attention(src_emb,padding_mask=src_mask)
    max_error=0.0
    for head_idx,weights in enumerate(attention_weights):
        row_sums=weights.sum(dim=-1)
        valid_queries=(src!=0)
        valid_sums=row_sums.masked_select(valid_queries)
        error=torch.abs(valid_sums-1.0).max().item()
        print(f"Head {head_idx+1}: max row-sum error = {error}")
        max_error=max(max_error,error)
    print("\nOverall maximum error:",max_error)
    if max_error<1e-6:
        print("ATTENTION ROW SUM CHECK: PASSED")
    else:
        print("ATTENTION ROW SUM CHECK: FAILED")
def weight_sharing_check(transformer):
    print("\nWEIGHT SHARING CHECK")
    embedding_weight=transformer.shared.emb.weight
    output_weight=transformer.output_projection.weight
    print("Embedding shape:",embedding_weight.shape)
    print("Output projection shape:",output_weight.shape)
    print("Same tensor:",embedding_weight is output_weight)
    print("Same storage:",embedding_weight.data_ptr()==output_weight.data_ptr())
    if embedding_weight is output_weight:
        print("WEIGHT SHARING CHECK: PASSED")
    else:
        print("WEIGHT SHARING CHECK: FAILED")
def get_lr(step,dmodel=256,warmup_steps=4000):
    return dmodel**(-0.5)*min(step**(-0.5),step*warmup_steps**(-1.5))
def learning_rate_schedule():
    print("\nLEARNING RATE SCHEDULE")
    steps=range(1,20001)
    learning_rates=[get_lr(step) for step in steps]
    os.makedirs("results",exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(steps,learning_rates)
    plt.axvline(4000,linestyle="--")
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Transformer Learning Rate Schedule")
    plt.grid(True)
    plt.savefig("results/learning_rate_schedule.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("Learning rate schedule saved.")
def gold_round_trip():
    print("\nGOLD ROUND-TRIP")
    gold_output="results/dev_gold_roundtrip.jsonl"
    with open("dev_pairs.jsonl",encoding="utf-8") as f,open(gold_output,"w",encoding="utf-8") as out:
        for line in f:
            pair=json.loads(line)
            parsed_query=parse_sql_string(pair["tgt"])
            out.write(json.dumps({"query":parsed_query})+"\n")
    print("Gold round-trip file created.")
    print("Lines:",sum(1 for _ in open(gold_output,encoding="utf-8")))
    os.system("python WikiSQL/evaluate.py WikiSQL/data/dev.jsonl WikiSQL/data/dev.db results/dev_gold_roundtrip.jsonl")
def cross_attention_map(transformer,dev_loader):
    print("\nCROSS-ATTENTION MAP")
    transformer.eval()
    src,tgt=next(iter(dev_loader))
    src=src[:1].to(device)
    src_mask=(src==0).unsqueeze(1)
    with torch.no_grad():
        memory=transformer.encoder(transformer.input_layer(src),src_mask)
        pred_tokens=greedy_decode(transformer,src,src_mask,max_len=64)
    decoder_tokens=pred_tokens[:,:-1]
    with torch.no_grad():
        x=transformer.input_layer(decoder_tokens)
        for layer_idx,layer in enumerate(transformer.decoder.layers):
            masked_output,_=layer.Multi_Head_Attention_1(x,masking=True,padding_mask=(decoder_tokens==0).unsqueeze(1))
            x=layer.layer_norm_1(x+layer.dropout(masked_output))
            cross_output,cross_weights=layer.Multi_Head_Attention_2(x,memory,padding_mask=src_mask)
            x=layer.layer_norm_2(x+layer.dropout(cross_output))
            ff_output=layer.Point_wise_feed_forward(x)
            x=layer.layer_norm_3(x+layer.dropout(ff_output))
        attention=torch.stack(cross_weights,dim=0)
    attention=attention.mean(dim=0)[0]
    generated_tokens=pred_tokens[0,1:].tolist()
    generated_text=sp.decode(generated_tokens).strip()
    source_tokens=sp.encode(sp.decode(src[0].tolist()),out_type=str)
    target_tokens=sp.encode(generated_text,out_type=str)
    attention=attention[:len(target_tokens),:len(source_tokens)]
    os.makedirs("results",exist_ok=True)
    plt.figure(figsize=(14,8))
    plt.imshow(attention.cpu().numpy(),aspect="auto")
    plt.xticks(range(len(source_tokens)),source_tokens,rotation=90,fontsize=7)
    plt.yticks(range(len(target_tokens)),target_tokens,fontsize=8)
    plt.xlabel("Source Tokens")
    plt.ylabel("Generated Tokens")
    plt.title("Decoder Cross-Attention Map - Last Layer")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/cross_attention_map.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("Cross-attention map saved.")
def query_equal(gold,pred):
    if "error" in pred:
        return False
    return gold["sel"]==pred["sel"] and gold["agg"]==pred["agg"] and set(map(tuple,gold["conds"]))==set(map(tuple,pred["conds"]))
def failure_type(gold,pred):
    if "error" in pred:
        return "Parse failure"
    errors=[]
    if gold["sel"]!=pred["sel"]:
        errors.append("Wrong column")
    if gold["agg"]!=pred["agg"]:
        errors.append("Wrong aggregation")
    gold_conds=set(map(tuple,gold["conds"]))
    pred_conds=set(map(tuple,pred["conds"]))
    if gold_conds!=pred_conds:
        gold_cols=set(c[0] for c in gold["conds"])
        pred_cols=set(c[0] for c in pred["conds"])
        gold_vals=set(str(c[2]) for c in gold["conds"])
        pred_vals=set(str(c[2]) for c in pred["conds"])
        if gold_cols!=pred_cols:
            errors.append("Missing/extra condition column")
        elif gold_vals!=pred_vals:
            errors.append("Wrong value")
        else:
            errors.append("Missing/extra condition")
    return ", ".join(errors) if errors else "Other"
def sql_from_query(query,headers):
    if "error" in query:
        return "Unparseable SQL output."
    sql="SELECT "
    if query["agg"]!=0:
        sql+=AGG_OPS[query["agg"]]+"("+headers[query["sel"]]+")"
    else:
        sql+=headers[query["sel"]]
    if query["conds"]:
        conditions=[]
        for col,op,val in query["conds"]:
            conditions.append(f"{headers[col]} {COND_OPS[op]} '{val}'")
        sql+=" WHERE "+" AND ".join(conditions)
    return sql
def qualitative_analysis():
    print("\nQUALITATIVE ANALYSIS")
    with open("WikiSQL/data/dev.jsonl",encoding="utf-8") as f:
        dev_data=[json.loads(line) for line in f]
    with open("WikiSQL/data/dev.tables.jsonl",encoding="utf-8") as f:
        tables={json.loads(line)["id"]:json.loads(line) for line in f}
    with open("results/dev_greedy.jsonl",encoding="utf-8") as f:
        predictions=[json.loads(line) for line in f]
    correct=[]
    wrong=[]
    for i,(example,prediction) in enumerate(zip(dev_data,predictions)):
        gold=example["sql"]
        table=tables[example["table_id"]]
        headers=table["header"]
        is_correct=query_equal(gold,prediction.get("query",prediction))
        item={"index":i,"question":example["question"],"gold":gold,"pred":prediction.get("query",prediction),"headers":headers}
        if is_correct and len(correct)<5:
            correct.append(item)
        elif not is_correct and len(wrong)<5:
            wrong.append(item)
    os.makedirs("results",exist_ok=True)
    with open("results/samples.md","w",encoding="utf-8") as f:
        f.write("# Qualitative Dev Samples\n\n")
        f.write("## Five Correct Examples\n\n")
        for n,item in enumerate(correct,1):
            f.write(f"### Correct Example {n}\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            f.write(f"**Gold SQL:** `{sql_from_query(item['gold'],item['headers'])}`\n\n")
            f.write(f"**Model SQL:** `{sql_from_query(item['pred'],item['headers'])}`\n\n")
            f.write("**Result:** Correct\n\n")
            f.write("---\n\n")
        f.write("## Five Wrong Examples\n\n")
        for n,item in enumerate(wrong,1):
            f.write(f"### Wrong Example {n}\n\n")
            f.write(f"**Question:** {item['question']}\n\n")
            f.write(f"**Gold SQL:** `{sql_from_query(item['gold'],item['headers'])}`\n\n")
            f.write(f"**Model SQL:** `{sql_from_query(item['pred'],item['headers'])}`\n\n")
            f.write(f"**Failure:** {failure_type(item['gold'],item['pred'])}\n\n")
            f.write("---\n\n")
    print("Qualitative samples saved.")
    print("Correct examples:",len(correct))
    print("Wrong examples:",len(wrong))
def official_test_evaluation():
    print("\nOFFICIAL WIKISQL TEST EVALUATION")
    official_evaluator("WikiSQL/data/test.jsonl","WikiSQL/data/test.db","results/test_beam.jsonl")
def run_all_evaluations():
    transformer,dev_loader,test_loader=load_model_and_data()
    dev_data_check(dev_loader)
    # write_predictions(transformer,dev_loader,output_file="results/dev_greedy.jsonl",use_beam=False)
    # write_predictions(transformer,dev_loader,output_file="results/dev_beam.jsonl",use_beam=True)
    # write_predictions(transformer,test_loader,output_file="results/test_beam.jsonl",use_beam=True)
    prediction_files_check()
    official_dev_evaluation()
    parse_failure_rate()
    component_accuracy()
    causal_mask_check(transformer,dev_loader)
    padding_mask_check(transformer,dev_loader)
    attention_row_sum_check(transformer,dev_loader)
    weight_sharing_check(transformer)
    learning_rate_schedule()
    gold_round_trip()
    cross_attention_map(transformer,dev_loader)
    qualitative_analysis()
    official_test_evaluation()
    print("\nALL EVALUATIONS COMPLETED.")
if __name__=="__main__":
    run_all_evaluations()
