import time
import pandas as pd
import numpy as np
from google.cloud import storage
from xgboost import XGBClassifier
from src.utils.conversation_utils import (compute_asr_len,get_flow_from_intent,get_suffix_from_intent)

class DetectOutOfPlaceOpensModel():
    def __init__(self, clean_asr_service, embeddings_service) -> None:
        self.clean_asr_service = clean_asr_service
        self.embeddings_service = embeddings_service
        self.model = XGBClassifier()

    def load_model(self, lang:str):
        print("<--------------LOADING MODEL!!!")
        time_start = time.time()
        model_path = self.download_model_from_bucket(lang)
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        time_end = time.time()
        print(f"Model loaded: {time_end - time_start} seconds")
        
    def download_model_from_bucket(self, lang: str): 
        model_filename = f"detect-out-of-place-open-responses-model-{lang}.json"
        client = storage.Client()
        bucket = client.get_bucket("ai-models-bucket")
        blob = bucket.blob(model_filename)
        model_local_path = "/tmp/model.json"
        blob.download_to_filename(model_local_path)
        return model_local_path
    
    def feature_preprocessing(self, intent: list, asr: list): 
        df = pd.DataFrame({"intent": intent, "asr": asr})

        #### Feature preprocessing ####
        df["asr_len"] = df.asr.apply(lambda x: compute_asr_len(x))
        df["flow"] = df.intent.apply(lambda x: get_flow_from_intent(x))
        df["suffix"] = df.intent.apply(lambda x: get_suffix_from_intent(x))
        df["cleaned_asr"] = df.asr.apply(lambda x: self.clean_asr_service.execute(x, delete_stopwords=False))
        df = self.compute_embeddings_features(df)
        df = self.handle_text_features(df)
        
        # Order features
        cols_when_model_builds = self.model.get_booster().feature_names
        X = df[cols_when_model_builds]
        return X

    def compute_embeddings_features(self, df): 
        df["embeddings"] = df["cleaned_asr"].apply(lambda x: self.embeddings_service.compute_embeddings(x, embedding_model="test-embeddings"))
        tags = df['embeddings'].apply(pd.Series)
        features = tags.rename(columns = lambda x : 'embedding_feature_' + str(x))
        result = pd.concat([df, features], axis=1)
        return result.drop(columns=["embeddings", "asr", "cleaned_asr", "intent"], axis=1)
    
    def handle_text_features(self, df): 
        # Extract text features -> convert to category
        cats = df.select_dtypes(exclude=np.number).columns.tolist()
        for col in cats:
            df[col] = df[col].astype('category')
        return df
    
    def predict(self, X) -> list:
        return self.model.predict(X)
    
    def handle_model_output(self, predictions: list, intents: list) -> list:
        # [True if pred == 1 else False for pred in predictions]
        output = []
        for intent,prediction in zip(intents,predictions):
            flow = get_flow_from_intent(intent)
            suffix = get_suffix_from_intent(intent)
            if prediction == 1 and flow != "bye" and suffix != "open" and suffix != "fallback":
                output.append(True)
            else:
                output.append(False)
        return output


    

