
from pathlib import Path
import re
import json
import hashlib
from datetime import datetime
import pandas as pd
import streamlit as st

from langchain_community.vectorstores import FAISS
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings


try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None 


try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    SystemMessage = None
    HumanMessage = None


# Config / paths

INDEX_DIR       = Path("indexes") / "safety_faiss"
EXPORT_XLSX     = Path("data") / "hazard_requirement_key_list.xlsx"  
VALIDATION_LOG  = Path("data") / "validation_log.xlsx"               
EMBED_MODEL     = "nomic-embed-text"

# Generation settings
GEN_ENABLED       = True          
GEN_MODEL         = "llama3"      
GEN_TEMPERATURE   = 0.2            
NUM_CANDIDATES    = 2              
MAX_REF_MESSAGES  = 3             




def load_retriever(k: int = 12):
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vs  = FAISS.load_local(str(INDEX_DIR), embeddings=emb, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})

def make_query(function: str, hazard: str, cause: str) -> str:
    return (
        f"Function: {function}\n"
        f"Hazard: {hazard}\n"
        f"Cause: {cause}\n"
        "Return concise safety messages."
    )


def get_hrk_from_meta(meta: dict) -> str | None:
    explicit = (meta.get("hazard_key") or "").strip()
    if explicit:
        return "{" + explicit.strip('{} ') + "}"
    F = (meta.get("F_code") or "").strip()
    G = (meta.get("G_code") or "").strip()
    U = (meta.get("U_code") or "").strip()
    M = (meta.get("M_code") or "").strip()
    if all([F, G, U, M]):
        return "{hz_ga_" + f"F{F}G{G}U{U}M{M}" + "}"
    return None




def get_meta(doc) -> dict:
    return doc.metadata if isinstance(doc.metadata, dict) else {}

def get_ref_text(doc) -> str:
    return (get_meta(doc).get("safety_message") or "").strip()

def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def exact_match(doc, f, h, c) -> bool:
    md = get_meta(doc)
    return (
        norm(md.get("function","")) == norm(f)
        and norm(md.get("hazard","")) == norm(h)
        and norm(md.get("cause","")) == norm(c)
    )

def dedupe_keep_first(ds, limit=2):
    seen, out = set(), []
    for d in ds:
        txt = (d.page_content or "").strip()
        if txt and txt not in seen:
            seen.add(txt)
            out.append(d)
        if len(out) == limit:
            break
    return out

def make_triplet_id(f: str, h: str, c: str) -> str:
    key = (norm(f) + "|" + norm(h) + "|" + norm(c)).encode("utf-8")
    return "TRI-" + hashlib.sha1(key).hexdigest()[:10]




def to_row(function, hazard, cause, safety_message, hrk):
    return {
        "Function": function,
        "Hazard": hazard,
        "Cause": cause,
        "SafetyMessage": safety_message,
        "HazardKey": hrk,
    }

def append_to_excel(row: dict, path: Path):
    cols = ["Function", "Hazard", "Cause", "SafetyMessage", "HazardKey"]
    df_new = pd.DataFrame([row], columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            df_old = pd.read_excel(path, engine="openpyxl")
        except PermissionError as e:
            raise RuntimeError(
                f"Cannot write to {path.name} because it appears to be open. "
                "Please close the Excel file and try again."
            ) from e
        except Exception:
            df_old = pd.DataFrame(columns=cols)
        for c in cols:
            if c not in df_old.columns:
                df_old[c] = ""
        df_old = df_old[cols]
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    try:
        df_out.to_excel(path, index=False, engine="openpyxl")
    except PermissionError as e:
        raise RuntimeError(
            f"Cannot write to {path.name} because it appears to be open. "
            "Please close the Excel file and try again."
        ) from e




def append_validation_log(triplet_id, f, h, c, gen_candidates, selected_idx, final_msg, hrk, ref_context):
    rows = {
        "timestamp":       [datetime.now().isoformat(timespec="seconds")],
        "triplet_id":      [triplet_id or ""],
        "function":        [f],
        "hazard":          [h],
        "cause":           [c],
        "generated_1":     [gen_candidates[0] if len(gen_candidates) > 0 else ""],
        "generated_2":     [gen_candidates[1] if len(gen_candidates) > 1 else ""],
        "selected_idx":    [selected_idx],
        "final_message":   [final_msg],
        "hazard_key":      [hrk or ""],
        "ref_context":     ["\n\n---\n".join(ref_context) if ref_context else ""],
    }
    df_new = pd.DataFrame(rows)
    VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    if VALIDATION_LOG.exists():
        try:
            df_old = pd.read_excel(VALIDATION_LOG, engine="openpyxl")
        except Exception:
            df_old = pd.DataFrame(columns=df_new.columns)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_excel(VALIDATION_LOG, index=False, engine="openpyxl")




def _clean(v: str) -> str:
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v

def parse_intent(text: str):
    t = text.strip()
    tl = t.lower()
    if tl.startswith("export"):
        return ("export", None)
    if tl.startswith("search"):
        m = re.search(r'function=(.*?)\s+hazard=(.*?)\s+cause=(.*)', t, re.IGNORECASE | re.DOTALL)
        if not m:
            m = re.search(r'function=(.*?)\s+hazard=(.*?)\s+cuase=(.*)', t, re.IGNORECASE | re.DOTALL)
        if m:
            data = {"function": _clean(m.group(1)),
                    "hazard":   _clean(m.group(2)),
                    "cause":    _clean(m.group(3))}
            return ("search", data)
        return ("needs_fields", {"function":"","hazard":"","cause":""})
    return ("chat", None)

 

def format_operator_manual_block(level: str, src: str, cons: str, cm: str) -> str:
    """Exact layout required for the visible/exported message."""
    level = (level or "CAUTION").strip().upper()
    if level not in {"CAUTION", "WARNING"}:
        level = "CAUTION"
    src = (src or "").strip()
    cons = (cons or "").strip()
    cm = (cm or "").strip()
    return (
        f"Operator Manual: {level}\n"
        f"Source: {src}\n"
        f"Consequence: {cons}\n"
        "Countermeasure:\n"
        f" {cm}"
    )


_REF_LEVEL_RE = re.compile(r"Operator\s*Manual[:\s-]*\s*(CAUTION|WARNING)", re.IGNORECASE)
_REF_SOURCE_RE = re.compile(r"^\s*Source\s*:?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_REF_CONS_RE1  = re.compile(r"^\s*Consequence\s*:?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_REF_CONS_RE2  = re.compile(r"^\s*Consequences\s*:?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_REF_CM_RE     = re.compile(r"^\s*Countermeasure\s*:?\s*(.*)$", re.IGNORECASE | re.MULTILINE | re.DOTALL)

def normalize_reference_text(raw: str) -> str:
    """
    If the reference already matches your block layout, return as-is.
    Else, try to extract level/source/consequence/countermeasure and format strictly.
    """
    text = (raw or "").strip()
    if not text:
        return ""

  
    if text.lower().startswith("operator manual"):
        return text

    
    level_m = _REF_LEVEL_RE.search(text)
    level = (level_m.group(1).upper() if level_m else "CAUTION")

    src_m = _REF_SOURCE_RE.search(text)
    cons_m = _REF_CONS_RE1.search(text) or _REF_CONS_RE2.search(text)
    cm_m = _REF_CM_RE.search(text)

    src = src_m.group(1).strip() if src_m else ""
    cons = cons_m.group(1).strip() if cons_m else ""
    cm = cm_m.group(1).strip() if cm_m else ""

    
    if not (src or cons or cm):
        return format_operator_manual_block(level, text, "", "")

    return format_operator_manual_block(level, src, cons, cm)




GEN_SYSTEM = (
    "You generate operator safety messages for a medical imaging system.\n"
    "HARD RULES:\n"
    "- Use Function/Hazard/Cause and any given reference excerpts as context.\n"
    "- Stay on topic. Do not invent device features or product names.\n"
    "- Return STRICT JSON only (no prose). Schema:\n"
    '{"messages":[{"level":"CAUTION or WARNING","Source":"...","Consequences":"...","Countermeasure":"..."}]}\n'
    "- Use the exact keys: level, Source, Consequences, Countermeasure.\n"
    "- Choose level=CAUTION or WARNING based on severity/urgency.\n"
    "- Keep text concise and operator-facing."
)

def _json_from(text: str) -> dict:
    """Parse JSON or extract the first JSON object block as fallback."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

def _normalize_generated_messages(data: dict, n: int) -> list[str]:
    """Normalize generated JSON into the exact block layout."""
    items = data.get("messages", []) if isinstance(data, dict) else []
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lvl  = (item.get("level") or item.get("Level") or "CAUTION").strip()
        src  = (item.get("Source") or item.get("source") or "").strip()
        cons = (item.get("Consequences") or item.get("Consequence") or "").strip()
        cm   = (item.get("Countermeasure") or item.get("countermeasure") or "").strip()
        out.append(format_operator_manual_block(lvl, src, cons, cm))
    # unique & limit
    seen, uniq = set(), []
    for m in out:
        if m and m not in seen:
            seen.add(m); uniq.append(m)
            if len(uniq) >= n:
                break
    return uniq

def generate_candidates(function: str, hazard: str, cause: str, references: list[str], n: int) -> list[str]:
    """
    Fallback generation ONLY if no exact reference is available.
    Returns list[str] in the strict block layout.
    """
    if not GEN_ENABLED or ChatOllama is None or n <= 0:
        return []

    refs_txt = ""
    if references:
        joined = "\n\n---\n".join(references[:MAX_REF_MESSAGES])
        refs_txt = f"\n\nReference safety messages:\n<<<\n{joined}\n>>>"

    user = (
        f"Function: {function}\n"
        f"Hazard: {hazard}\n"
        f"Cause: {cause}\n"
        f"Create up to {n} operator-facing safety message(s) with level/Source/Consequences/Countermeasure."
        f"{refs_txt}"
    )

    try:
        llm = ChatOllama(model=GEN_MODEL, temperature=GEN_TEMPERATURE, format="json")
    except TypeError:
        llm = ChatOllama(model=GEN_MODEL, temperature=GEN_TEMPERATURE)

    try:
        if SystemMessage and HumanMessage:
            rsp = llm.invoke([SystemMessage(content=GEN_SYSTEM), HumanMessage(content=user)])
        else:
            prompt = f"<<SYS>>\n{GEN_SYSTEM}\n<</SYS>>\n{user}"
            rsp = llm.invoke(prompt)
        content = rsp.content if hasattr(rsp, "content") else str(rsp)
        data = _json_from(content)
        msgs = _normalize_generated_messages(data, n)
        return msgs
    except Exception as e:
        try:
            print("[GENERATION ERROR]", repr(e))
            print("[RAW MODEL OUTPUT]\n", content)
        except Exception:
            print("[GENERATION ERROR]", repr(e))
        return []




from PIL import Image

st.set_page_config(page_title="Safety Chatbot", page_icon="🤖", layout="centered")


st.markdown("""
    <style>
        /* Make the chat input box larger */
        div[data-testid="stChatInput"] textarea {
            min-height: 80px;       /* increase height */
            font-size: 16px;        /* larger text */
            padding: 12px;
        }
        /* Disable browser spellcheck underline */
        div[data-testid="stChatInput"] textarea {
            spellcheck: false;
            -webkit-text-size-adjust: none;
        }
        /* Optional: border color tweak for a cleaner look */
        div[data-testid="stChatInput"] textarea:focus {
            border: 2px solid #FF6600 !important;  /* Siemens orange */
            outline: none !important;
        }
    </style>
""", unsafe_allow_html=True)



logo = Image.open("images/siemens_logo.png")

col1, col2 = st.columns([4, 2])  # adjust spacing: 6 parts for title, 1 for logo
with col1:
    st.title("🤖 Safety Chatbot")
with col2:
    st.image(logo, use_container_width=True)  # You can increase/decrease width if needed
# Session state
if "messages" not in st.session_state: st.session_state.messages = []
if "generated" not in st.session_state: st.session_state.generated = []          # list[str]
if "last_inputs" not in st.session_state: st.session_state.last_inputs = None
if "selected_idx" not in st.session_state: st.session_state.selected_idx = None
if "selected_hrk" not in st.session_state: st.session_state.selected_hrk = None
if "selection_confirmed" not in st.session_state: st.session_state.selection_confirmed = False
if "triplet_id" not in st.session_state: st.session_state.triplet_id = None
if "final_message" not in st.session_state: st.session_state.final_message = ""
if "ref_context" not in st.session_state: st.session_state.ref_context = []     # list[str]

with st.sidebar:
    st.markdown("### Mode: Reference-Strict")
    st.markdown(
        "1) `search function=... hazard=... cause=...`\n"
        "2) If an **exact** reference exists → (no LLM).\n"
        "3) You can still **Select → (Edit) → Confirm**.\n"
        "4) If **no exact reference** exists → we cautiously generate with strict formatting.\n"
        "5) Type `export` to append the 5 columns to Excel."
    )
    st.caption(f"Excel path: {EXPORT_XLSX}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input(
    "Try: search function=Scanner Movement and positioning hazard=Cuts and other injuries cause=Patient improperly positioned where front cover impacts shoulders"
)
if user_text:
    st.session_state.messages.append({"role":"user", "content":user_text})
    intent, payload = parse_intent(user_text)

    if intent == "needs_fields":
        reply = "Please provide function, hazard, and cause. Example:\n`search function=patient registration (POF) hazard=incorrect diagnosis basis cause=wrong entry of patient orientation`"
        with st.chat_message("assistant"): st.markdown(reply)
        st.session_state.messages.append({"role":"assistant","content":reply})

    elif intent == "search":
        f = payload["function"]; h = payload["hazard"]; c = payload["cause"]
        with st.chat_message("assistant"):
            st.markdown("Retrieving references…")
            retriever = load_retriever(k=12)
            docs = retriever.get_relevant_documents(make_query(f, h, c))

            
            exact_docs = [d for d in docs if exact_match(d, f, h, c)]
            refs = dedupe_keep_first(exact_docs or docs, limit=MAX_REF_MESSAGES)

           
            ref_texts = [get_ref_text(d) for d in refs if get_ref_text(d)]
            st.session_state.ref_context = ref_texts

         
            best_meta = get_meta(refs[0]) if refs else {}

            if exact_docs:
                normalized = []
                for d in dedupe_keep_first(exact_docs, limit=NUM_CANDIDATES):
                    ref_raw = get_ref_text(d)
                    normalized.append(normalize_reference_text(ref_raw))
                st.session_state.last_inputs = (f, h, c)
                st.session_state.generated = normalized
                st.session_state.selected_idx = None
                st.session_state.selection_confirmed = False
                st.session_state.triplet_id = make_triplet_id(f, h, c)
                st.session_state.final_message = ""
                st.session_state._best_meta_for_hrk = best_meta
                st.markdown(f"Found **exact reference**. Showing {len(normalized)} option(s) from reference (no generation).")
            else:
                
                gen = generate_candidates(f, h, c, ref_texts, NUM_CANDIDATES)
                if not gen:
                    st.markdown("No messages could be generated. Adjust your inputs or check the local model.")
                else:
                    st.session_state.last_inputs = (f, h, c)
                    st.session_state.generated = gen[:NUM_CANDIDATES]
                    st.session_state.selected_idx = None
                    st.session_state.selection_confirmed = False
                    st.session_state.triplet_id = make_triplet_id(f, h, c)
                    st.session_state.final_message = ""
                    st.session_state._best_meta_for_hrk = best_meta
                    st.markdown(f"No exact reference found. Generated **{len(st.session_state.generated)}** option(s).")

    elif intent == "export":
        with st.chat_message("assistant"):
            gen = st.session_state.generated
            idx  = st.session_state.selected_idx
            finputs = st.session_state.last_inputs
            hrk = st.session_state.selected_hrk
            final_msg = st.session_state.final_message
            if not gen or idx is None or not finputs or not hrk or not final_msg:
                st.markdown("Nothing to export yet. Run a `search …`, pick one option, **edit if needed**, then **Confirm selection** first.")
            else:
                f, h, c = finputs
                row = to_row(f, h, c, final_msg, hrk)
                try:
                    append_to_excel(row, EXPORT_XLSX)
                except RuntimeError as e:
                    st.error(str(e))
                else:
                    st.markdown("Exported **1** row (appended).")
                    st.caption(f"Excel: {EXPORT_XLSX.resolve()}")

    else:
        reply = "Use: `search function=... hazard=... cause=...` — I retrieve references and (only if needed) generate."
        with st.chat_message("assistant"): st.markdown(reply)
        st.session_state.messages.append({"role":"assistant","content":reply})


if st.session_state.generated:
    st.markdown("### Safety Message Options")
    if st.session_state.triplet_id:
        st.caption(f"Triplet ID: {st.session_state.triplet_id}")

    # Render options
    for i, text in enumerate(st.session_state.generated, 1):
        with st.expander(f"Option {i}", expanded=True):
            st.write(text)
            if st.button("Select this option", key=f"pick_{i}"):
                st.session_state.selected_idx = i - 1
                st.session_state.selection_confirmed = False
                st.session_state.selected_hrk = None
                st.session_state.final_message = text  # default to displayed text

   
    if st.session_state.selected_idx is None:
        st.info("No option selected yet.")
    else:
        st.success(f"Currently selected: Option {st.session_state.selected_idx + 1}")
        default_msg = st.session_state.final_message or st.session_state.generated[st.session_state.selected_idx]
        st.session_state.final_message = st.text_area(
            "Edit safety message (optional):",
            value=default_msg,
            key="edited_msg",
            height=220
        )


    if st.button("Confirm selection", key="confirm_btn"):
        if st.session_state.selected_idx is None:
            st.warning("Pick an option first.")
        else:
            st.session_state.selection_confirmed = True
            best_meta = getattr(st.session_state, "_best_meta_for_hrk", {}) or {}
            st.session_state.selected_hrk = get_hrk_from_meta(best_meta)
            if not st.session_state.selected_hrk:
                st.warning("Cannot generate hazard key (codes missing from references). Provide codes or refine inputs.")

           
            if st.session_state.last_inputs:
                f, h, c = st.session_state.last_inputs
            else:
                f = h = c = ""
            append_validation_log(
                st.session_state.triplet_id,
                f, h, c,
                st.session_state.generated,
                st.session_state.selected_idx,
                st.session_state.final_message or st.session_state.generated[st.session_state.selected_idx],
                st.session_state.selected_hrk,
                st.session_state.ref_context
            )

  
    if st.session_state.selection_confirmed and st.session_state.selected_idx is not None:
        st.markdown("###  Selected")
        st.write(st.session_state.final_message)
        if st.session_state.selected_hrk:
            st.markdown(f"**Hazard Key:** `{st.session_state.selected_hrk}`")
        else:
            st.warning("HRK not found and cannot be generated (codes missing).")
        st.caption("Type `export` to save 5 columns (Function, Hazard, Cause, SafetyMessage, HazardKey).")
    else:
        st.caption("Select an option, optionally edit, then press **Confirm selection**.")
