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


def beam_search_decode(model, src, src_mask=None, beam_size=4, max_len=64):
    model.eval()
    device = src.device
    memory = model.encoder(model.input_layer(src),src_mask)
    tgt = torch.full((1,1),BOS_ID,dtype=torch.long,device=device)
    beams = [(tgt,0.0)]

    for _ in range(max_len):
        new_beams = []

        for seq,score in beams:
            if seq[0,-1].item() == EOS_ID:
                new_beams.append((seq,score))
                continue

            tgt_padding_mask = (seq == 0).unsqueeze(1)
            decoder_input = model.input_layer(seq)
            out = model.decoder(decoder_input,memory,tgt_padding_mask,src_mask)
            out = out[:,-1,:]
            prob = torch.log_softmax(model.output_projection(out),dim=-1)
            top_probs, top_ix = prob.topk(beam_size,dim=1)

            for i in range(beam_size):
                next_token = top_ix[0,i].view(1,1)
                new_seq = torch.cat([seq,next_token],dim=1)
                new_score = score + top_probs[0,i].item()
                new_beams.append((new_seq,new_score))

        beams = sorted(new_beams,key=lambda x:x[1],reverse=True)[:beam_size]

        if all(seq[0,-1].item() == EOS_ID for seq,_ in beams):
            break

    return beams[0][0]


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

    with open(output_file,"w",encoding="utf-8") as f:
        with torch.no_grad():

            for batch in dataloader:
                src,tgt = batch
                src = src.to(next(model.parameters()).device)
                src_mask = (src == 0).unsqueeze(1)

                for i in range(src.size(0)):
                    single_src = src[i].unsqueeze(0)
                    single_mask = src_mask[i].unsqueeze(0)

                    if use_beam:
                        pred_tokens = beam_search_decode(model,single_src,single_mask)
                    else:
                        pred_tokens = greedy_decode(model,single_src,single_mask)

                    decoded_str = sp.decode(pred_tokens[0].tolist())
                    decoded_str = decoded_str.replace("<s>","").replace("</s>","").strip()
                    parsed_query = parse_sql_string(decoded_str)

                    f.write(json.dumps({"query":parsed_query}) + "\n")


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