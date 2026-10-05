import re
import json
import torch
import sentencepiece as spm

AGG_OPS = ["","MAX","MIN","COUNT","SUM","AVG"]
COND_OPS = ["=",">","<"]

sp = spm.SentencePieceProcessor(model_file="sql_sp.model")

BOS_ID, EOS_ID = 2, 3


def greedy_decode(model, src, src_mask=None, max_len=64):
    model.eval()
    device = src.device
    memory = model.encoder(model.input_layer(src),src_mask)
    tgt = torch.full((src.size(0),1),BOS_ID,dtype=torch.long,device=device)

    for _ in range(max_len):
        tgt_padding_mask = (tgt == 0).unsqueeze(1)
        decoder_input = model.input_layer(tgt)
        out = model.decoder(decoder_input,memory,tgt_padding_mask,src_mask)
        out = out[:,-1,:]
        prob = model.output_projection(out)
        _, next_word = torch.max(prob,dim=1)
        tgt = torch.cat([tgt,next_word.unsqueeze(1)],dim=1)

        if (next_word == EOS_ID).all():
            break

    return tgt


def beam_search_decode(model,src,src_mask=None,beam_size=4,max_len=64):
    model.eval()
    device=src.device
    batch_size=src.size(0)

    memory=model.encoder(model.input_layer(src),src_mask)
    memory=memory.unsqueeze(1).expand(batch_size,beam_size,memory.size(1),memory.size(2))
    memory=memory.reshape(batch_size*beam_size,memory.size(2),memory.size(3))

    src_mask=src_mask.unsqueeze(1).expand(batch_size,beam_size,*src_mask.shape[1:])
    src_mask=src_mask.reshape(batch_size*beam_size,*src_mask.shape[2:])

    sequences=torch.full((batch_size*beam_size,1),BOS_ID,dtype=torch.long,device=device)
    scores=torch.zeros(batch_size,beam_size,device=device)
    scores[:,1:]=-float("inf")
    scores=scores.view(-1)
    finished=torch.zeros(batch_size*beam_size,dtype=torch.bool,device=device)

    for _ in range(max_len):
        tgt_padding_mask=(sequences==0).unsqueeze(1)
        decoder_input=model.input_layer(sequences)
        out=model.decoder(decoder_input,memory,tgt_padding_mask,src_mask)
        logits=torch.log_softmax(model.output_projection(out[:,-1,:]),dim=-1)

        logits[finished,:]=-float("inf")
        logits[finished,EOS_ID]=0.0

        vocab_size=logits.size(-1)
        candidate_scores=(scores.unsqueeze(1)+logits).view(batch_size,beam_size*vocab_size)
        top_scores,top_indices=torch.topk(candidate_scores,beam_size,dim=1)

        beam_indices=top_indices//vocab_size
        token_indices=top_indices%vocab_size

        offsets=(torch.arange(batch_size,device=device)*beam_size).unsqueeze(1)
        global_indices=(beam_indices+offsets).view(-1)

        sequences=sequences[global_indices]
        memory=memory[global_indices]
        src_mask=src_mask[global_indices]
        finished=finished[global_indices]

        sequences=torch.cat([sequences,token_indices.view(-1,1)],dim=1)
        scores=top_scores.view(-1)
        finished=finished|(token_indices.view(-1)==EOS_ID)

        if finished.view(batch_size,beam_size).all():
            break

    best_indices=scores.view(batch_size,beam_size).argmax(dim=1)
    offsets=torch.arange(batch_size,device=device)*beam_size
    best_indices=best_indices+offsets

    return sequences[best_indices]


def parse_sql_string(decoded_str):
    try:
        parsed = {"sel":0,"agg":0,"conds":[]}
        parts = decoded_str.split("where")
        select_clause = parts[0].strip()
        where_clause = parts[1].strip() if len(parts) > 1 else ""

        for i,agg in enumerate(AGG_OPS):
            if agg != "" and agg.lower() in select_clause.lower():
                parsed["agg"] = i
                break

        sel_col_match = re.search(r'<c(\d+)>',select_clause)

        if sel_col_match:
            parsed["sel"] = int(sel_col_match.group(1))

        if where_clause:
            conditions = where_clause.split(" and ")

            for cond in conditions:
                col_match = re.search(r'<c(\d+)>',cond)

                if not col_match:
                    continue

                col_idx = int(col_match.group(1))
                op_idx = 0
                val_str = ""

                for i,op in enumerate(COND_OPS):
                    if op in cond:
                        op_idx = i
                        val_str = cond.split(op,1)[1].strip()
                        break

                parsed["conds"].append([col_idx,op_idx,val_str])

        return parsed

    except Exception:
        return {"error":"parse"}


def write_predictions(model,dataloader,output_file="test_output.jsonl",use_beam=False):
    model.eval()
    device=next(model.parameters()).device
    total=len(dataloader.dataset)
    count=0

    with open(output_file,"w",encoding="utf-8") as f:
        with torch.no_grad():
            for batch in dataloader:
                src,tgt=batch
                src=src.to(device)
                src_mask=(src==0).unsqueeze(1)

                if use_beam:
                    pred_tokens=beam_search_decode(model,src,src_mask,beam_size=4,max_len=64)
                else:
                    pred_tokens=greedy_decode(model,src,src_mask,max_len=64)

                for i in range(src.size(0)):
                    tokens=pred_tokens[i].tolist()

                    if EOS_ID in tokens:
                        tokens=tokens[:tokens.index(EOS_ID)+1]

                    decoded_str=sp.decode(tokens)
                    decoded_str=decoded_str.replace("<s>","").replace("</s>","").strip()
                    parsed_query=parse_sql_string(decoded_str)

                    f.write(json.dumps({"query":parsed_query})+"\n")

                    count+=1

                    if use_beam and (count%10==0 or count==total):
                        print(f"Beam progress: {count}/{total}",flush=True)


def readable_sql(parsed_query,column_names):
    if "error" in parsed_query:
        return "Unparseable SQL output."

    sel_col = column_names[parsed_query.get("sel",0)]
    agg_idx = parsed_query.get("agg",0)
    agg_op = AGG_OPS[agg_idx] if agg_idx < len(AGG_OPS) else ""

    if agg_op:
        sql = f"SELECT {agg_op}({sel_col})"
    else:
        sql = f"SELECT {sel_col}"

    conds = parsed_query.get("conds",[])

    if conds:
        sql += " WHERE "
        cond_strings = []

        for cond in conds:
            col_name = column_names[cond[0]]
            op = COND_OPS[cond[1]]
            val = cond[2]
            cond_strings.append(f"{col_name} {op} '{val}'")

        sql += " AND ".join(cond_strings)

    return sql