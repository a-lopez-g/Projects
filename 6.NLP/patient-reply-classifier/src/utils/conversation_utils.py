def get_suffix_from_intent(intent: str) -> str:
    return norm_str(intent.split("-")[-1].strip())

def get_flow_from_intent(intent: str) -> str:
    return norm_str(intent.split("-")[0].strip())

def norm_str(x:str) -> str:
    return x.lower().strip()

def compute_asr_len(asr: str) -> int:
    try: 
        n_words = len(set(norm_str(asr).split()))
    # Si está vacío el asr
    except:
        n_words = 0
    return n_words