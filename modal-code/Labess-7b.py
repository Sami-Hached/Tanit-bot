import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "linagora/Labess-7b-chat"

# MPS on Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# On macOS, fp16 on MPS is typically the practical choice.
# If you see stability issues, switch torch_dtype to float32 (slower).
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "mps" else None,
    device_map={"": device},  # put the whole model on MPS if possible
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt_ar = (
    "يمكنك الإجابة باللهجة التونسية فقط.\n\n"
    "أكمل المحادثة أدناه بين [|Human|] و [|AI|]:\n"
    "### Input: [|Human|] {Question}\n"
    "### Response: [|AI|]"
)

@torch.inference_mode()
def get_response(question: str):
    text = prompt_ar.format_map({"Question": question})
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=128,          # better than max_length here
        do_sample=True,
        top_p=0.9,
        temperature=0.3,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Your model template uses "### Response:" (note colon). Keep robust parsing.
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    if "### Response :" in decoded:
        return decoded.split("### Response :")[-1].strip()
    return decoded.strip()

print(get_response("آش نقصدو كي نقولو لاباس"))
