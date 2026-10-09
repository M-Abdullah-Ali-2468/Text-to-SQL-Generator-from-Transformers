import os
import sys
import re

# 1. Streamlit page configuration (MUST be the first Streamlit command executed)
import streamlit as st

st.set_page_config(
    page_title="Text-to-SQL Transformer",
    page_icon="🗄️",
    layout="wide"
)

# 2. Path configuration: Ensure current working directory and sys.path point to the repository root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

try:
    os.chdir(parent_dir)
except Exception:
    pass

for path in [parent_dir, os.path.join(parent_dir, "starter"), current_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 3. Core dependencies
import torch
import sentencepiece as spm

# 4. Safe imports with fallbacks
import_errors = []

try:
    from model.transformers import Transformers
except Exception as e:
    import_errors.append(f"Model architecture import failed: {e}")

try:
    from starter.data_prep import encode_source
except Exception:
    try:
        from data_prep import encode_source
    except Exception as e:
        import_errors.append(f"Data prep import failed: {e}")

try:
    from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID
except Exception:
    try:
        from tokenizer import PAD_ID, BOS_ID, EOS_ID
    except Exception:
        PAD_ID, BOS_ID, EOS_ID = 0, 2, 3

try:
    from decoding import greedy_decode, beam_search_decode, parse_sql_string, readable_sql, AGG_OPS, COND_OPS
except Exception:
    AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    COND_OPS = ["=", ">", "<"]

    def greedy_decode(model, src, src_mask=None, max_len=64):
        model.eval()
        device = src.device
        memory = model.encoder(model.input_layer(src), src_mask)
        tgt = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_padding_mask = (tgt == 0).unsqueeze(1)
            decoder_input = model.input_layer(tgt)
            out = model.decoder(decoder_input, memory, tgt_padding_mask, src_mask)
            out = out[:, -1, :]
            prob = model.output_projection(out)
            _, next_word = torch.max(prob, dim=1)
            tgt = torch.cat([tgt, next_word.unsqueeze(1)], dim=1)
            if (next_word == EOS_ID).all():
                break
        return tgt

    beam_search_decode = None
    parse_sql_string = None
    readable_sql = None

if import_errors:
    for err in import_errors:
        st.error(f"❌ {err}")
    st.info("Please verify that all project dependencies and files are in the repository.")
    st.stop()


# ==========================================
# CACHED MODEL & TOKENIZER LOADING
# ==========================================
@st.cache_resource
def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Locate and load SentencePiece Tokenizer
    sp_candidates = [
        os.path.join(parent_dir, "sql_sp.model"),
        os.path.join(parent_dir, "starter", "sql_sp.model"),
        "sql_sp.model"
    ]
    sp_path = None
    for p in sp_candidates:
        if os.path.exists(p):
            sp_path = p
            break

    if not sp_path:
        raise FileNotFoundError("Could not find 'sql_sp.model' tokenizer file.")

    sp = spm.SentencePieceProcessor(model_file=sp_path)

    # 2. Instantiate Model Architecture (matching trained checkpoint)
    # Architecture: enc_layers=3, dec_layers=3, dmodel=256, heads=4, dff=1024, dropout=0.1
    model = Transformers(
        enc_layers=3,
        dec_layers=3,
        dmodel=256,
        heads=4,
        dff=1024,
        dropout=0.1
    )

    # 3. Locate and load trained weights
    ckpt_candidates = [
        os.path.join(parent_dir, "checkpoints", "best_transformer.pt"),
        os.path.join(parent_dir, "checkpoints", "best_model.pt"),
        os.path.join(parent_dir, "results", "best_model.pt"),
        os.path.join(parent_dir, "best_transformer.pt"),
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break

    metadata = {
        "device": str(device),
        "vocab_size": sp.get_piece_size(),
        "checkpoint_loaded": False,
        "epoch": None,
        "val_loss": None,
        "path": ckpt_path
    }

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            metadata["epoch"] = checkpoint.get("epoch")
            metadata["val_loss"] = checkpoint.get("validation_loss")
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        metadata["checkpoint_loaded"] = True
    else:
        st.warning("⚠️ Checkpoint file not found. Running with uninitialized weights.")

    model.to(device)
    model.eval()
    return model, sp, device, metadata


# ==========================================
# SQL POST-PROCESSING & FORMATTING
# ==========================================
def format_sql(raw_str, header_columns):
    """Converts the raw model token sequence into a clean, human-readable SQL query."""
    if parse_sql_string is not None and readable_sql is not None:
        try:
            parsed = parse_sql_string(raw_str)
            if "error" not in parsed:
                sel_idx = parsed.get("sel", 0)
                if 0 <= sel_idx < len(header_columns):
                    clean_query = readable_sql(parsed, header_columns)
                    # Improve query readability
                    formatted = clean_query.replace(" WHERE ", "\nWHERE ").replace(" AND ", "\n  AND ")
                    return formatted, parsed
        except Exception:
            pass

    # Fallback substitution if structured AST parsing fails
    readable = raw_str.strip()
    for i, col in enumerate(header_columns):
        readable = re.sub(rf"<c{i}>", f'"{col}"', readable, flags=re.IGNORECASE)

    # Standardize capitalization
    readable = re.sub(r'\bselect\b', 'SELECT', readable, flags=re.IGNORECASE)
    readable = re.sub(r'\bwhere\b', '\nWHERE', readable, flags=re.IGNORECASE)
    readable = re.sub(r'\band\b', '\n  AND', readable, flags=re.IGNORECASE)
    return readable, {"raw": raw_str}


# ==========================================
# INFERENCE PIPELINE
# ==========================================
def generate_sql(model, sp, device, question, header_columns, use_beam=False, beam_size=4):
    """Encodes input question & columns, performs decoding, and returns formatted SQL."""
    src_text = encode_source(question, header_columns)
    src_ids = sp.encode(src_text) + [EOS_ID]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src_tensor == 0).unsqueeze(1)

    with torch.no_grad():
        if use_beam and beam_search_decode is not None:
            pred_tokens = beam_search_decode(model, src_tensor, src_mask, beam_size=beam_size, max_len=64)
        else:
            pred_tokens = greedy_decode(model, src_tensor, src_mask, max_len=64)

        tokens = pred_tokens[0].tolist()
        if EOS_ID in tokens:
            tokens = tokens[:tokens.index(EOS_ID) + 1]

    raw_prediction = sp.decode(tokens).replace("<s>", "").replace("</s>", "").strip()
    readable_query, parsed_data = format_sql(raw_prediction, header_columns)
    return raw_prediction, readable_query, parsed_data, src_text


# ==========================================
# USER INTERFACE
# ==========================================
st.title("🗄️ Text-to-SQL Generator")
st.caption("Translate natural language questions into executable SQL queries using a Seq2Seq Transformer built from scratch.")

# Load system resources
try:
    model, sp, device, meta = load_system()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# Sidebar: Model status & settings
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    st.markdown("**Status:** " + ("🟢 Checkpoint Active" if meta["checkpoint_loaded"] else "🟡 Default Weights"))
    st.text(f"Device: {meta['device'].upper()}")
    st.text(f"Vocab Size: {meta['vocab_size']}")
    if meta.get("epoch") is not None:
        st.text(f"Epoch: {meta['epoch']}")
    if meta.get("val_loss") is not None:
        st.text(f"Validation Loss: {meta['val_loss']:.4f}")

    st.divider()
    st.subheader("Decoding Strategy")
    decoding_method = st.radio("Method", ["Greedy Search", "Beam Search"], index=0)
    use_beam = (decoding_method == "Beam Search")
    beam_size = 4
    if use_beam:
        beam_size = st.slider("Beam Width", min_value=2, max_value=8, value=4, step=1)

    st.divider()
    st.subheader("Preset Examples")
    preset_options = {
        "Custom": None,
        "Player Nationality": {
            "question": "What is Terrence Ross' nationality?",
            "columns": "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
        },
        "Jersey Number #42": {
            "question": "Who is the player that wears number 42?",
            "columns": "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
        },
        "Rockets Years": {
            "question": "Which player who played for the Rockets for the years 1986-92?",
            "columns": "Player, Years for Rockets, Position, Nationality"
        },
        "Earliest Elected": {
            "question": "What is the earliest years any of the incumbents were first elected?",
            "columns": "District, Incumbent, Party, First elected, Result"
        }
    }
    selected_preset = st.selectbox("Choose a sample query:", list(preset_options.keys()))

# Default input values based on preset
default_q = "What is Terrence Ross' nationality?"
default_cols = "Player, No., Nationality, Position, Years in Toronto, School/Club Team"

if selected_preset != "Custom" and preset_options[selected_preset] is not None:
    default_q = preset_options[selected_preset]["question"]
    default_cols = preset_options[selected_preset]["columns"]

# Main Query Formulation Form
with st.form("sql_query_form"):
    st.subheader("Query Parameters")
    user_question = st.text_input(
        "Natural Language Question:",
        value=default_q,
        placeholder="e.g. Which player scored more than 30 points?"
    )
    user_columns = st.text_input(
        "Table Columns (comma-separated):",
        value=default_cols,
        placeholder="e.g. Player, Points, Rebounds, Team"
    )
    
    submitted = st.form_submit_button("Generate SQL", type="primary")

if submitted:
    if not user_question.strip() or not user_columns.strip():
        st.warning("Please provide both a natural language question and table column headers.")
    else:
        columns = [col.strip() for col in user_columns.split(",") if col.strip()]
        if not columns:
            st.error("Please provide at least one valid column name.")
        else:
            with st.spinner("Generating SQL query..."):
                try:
                    raw_sql, clean_sql, parsed_ast, encoded_src = generate_sql(
                        model=model,
                        sp=sp,
                        device=device,
                        question=user_question,
                        header_columns=columns,
                        use_beam=use_beam,
                        beam_size=beam_size
                    )

                    st.success("Query generated successfully!")
                    st.subheader("Generated SQL Query")
                    st.code(clean_sql, language="sql")

                    with st.expander("🔍 Model Details & Internal Representation"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Raw Model Output:**")
                            st.code(raw_sql, language="text")
                            st.markdown("**Encoded Source Input:**")
                            st.code(encoded_src, language="text")
                        with col2:
                            st.markdown("**Parsed Structure (AST):**")
                            st.json(parsed_ast)

                except Exception as ex:
                    st.error(f"Inference error: {ex}")