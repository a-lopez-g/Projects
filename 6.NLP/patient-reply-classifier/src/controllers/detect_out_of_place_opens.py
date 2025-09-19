from src.usecases.index import detect_out_of_place_opens_usecase


def detect_out_of_place_opens_controller(request):
    intent = request["intent"]
    asr = request["asr"]
    lang_code = request["lang"]
    lang = lang_code.split("-")[0]

    if lang in ["es","en"]:
        result = detect_out_of_place_opens_usecase.execute(intent, asr, lang)
    else:
        result = [False]*len(intent)
    return result
