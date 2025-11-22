
import json, re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

GEN_MODEL = "llama3"   
GEN_TEMPERATURE = 0.0  

GEN_SYSTEM = (
    "You generate operator safety messages for a medical imaging system.\n"
    "HARD RULES:\n"
    "- Base content ONLY on the given Function, Hazard, Cause (and any references if provided).\n"
    "- Stay on topic. Do not invent device features/names.\n"
    "- Return STRICT JSON only (no extra text). Schema:\n"
    '{"messages":[{"level":"CAUTION or WARNING","Source":"...","Consequences":"...","Countermeasure":"..."}]}\n'
    "- Use the exact keys: level, Source, Consequences, Countermeasure.\n"
    "- Choose level=CAUTION or WARNING based on how severe/urgent the Consequences are.\n"
    "- Keep text concise and operator-facing."
)

USER_PROMPT = (
    "Function: Patient registration (POF)\n"
    "Hazard: Incorrect diagnosis basis\n"
    "Cause: Wrong entry of patient orientation\n"
    "Create 1 operator-facing safety message with Source/Consequences/Countermeasure and a level."
)

def _json_from(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

def _format_block(level: str, src: str, cons: str, cm: str) -> str:
    level = (level or "CAUTION").strip().upper()
    src   = (src  or "").strip()
    cons  = (cons or "").strip()
    cm    = (cm   or "").strip()
    return (
        f"Operator Manual:{level}\n"
        f"Source:{src}\n"
        f"Consequences:{cons}\n"
        "Countermeasure:\n"
        f"{cm}"
    )

def main():
    llm = ChatOllama(model=GEN_MODEL, temperature=GEN_TEMPERATURE, format="json")
    rsp = llm.invoke([SystemMessage(content=GEN_SYSTEM), HumanMessage(content=USER_PROMPT)])
    raw = rsp.content if hasattr(rsp, "content") else str(rsp)

    print("=== RAW MODEL OUTPUT ===")
    print(raw)

    data = _json_from(raw)
    items = data.get("messages", []) if isinstance(data, dict) else []
    if not items:
        print("\n No 'messages' in JSON. Double-check the daemon/model and system prompt.")
        return

    print("\n=== NORMALIZED MESSAGE(S) ===")
    for i, item in enumerate(items, 1):
        lvl = (item.get("level") or "CAUTION or WARNING").strip()
        src = (item.get("Source") or item.get("source") or "").strip()
        con = (item.get("Consequences") or item.get("Consequence") or "").strip()
        cm  = (item.get("Countermeasure") or item.get("countermeasure") or "").strip()
        print(f"\nOption {i}:\n{_format_block(lvl, src, con, cm)}")

if __name__ == "__main__":
    main()
