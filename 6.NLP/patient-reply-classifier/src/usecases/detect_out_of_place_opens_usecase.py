
class DetectOutOfPlaceOpensUsecase():
    def __init__(self, clasification_model) -> None:
        self.clasification_model = clasification_model
    
    def execute(self, intent: list, asr: list, lang: str): 
        # Load model
        self.clasification_model.load_model(lang)
        # Clean asr, embeddings, flow, suffix, order features
        X = self.clasification_model.feature_preprocessing(intent,asr)
        # Predict -> 0: No open, 1: Open 
        predictions = self.clasification_model.predict(X)
        # Boolean array prediction
        is_out_of_place_open = self.clasification_model.handle_model_output(predictions,intent)

        return is_out_of_place_open
