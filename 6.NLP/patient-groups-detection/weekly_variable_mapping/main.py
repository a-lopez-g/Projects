import firebase_admin
from firebase_admin import firestore, credentials
from variable_mapper import ConversationToVariableMapper as conversationMapper
import os
from io import StringIO
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

def weekly_variable_mapping(client_id,protocol_id):

    #request_json = request.get_json()
    #client_id = request_json["client_id"]
    #protocol_id = request_json["protocol_id"]

    mapping_file = db_dev.collection("disease-insight-mappings").document(protocol_id).collection("variable-mappings").document("latest").get().to_dict()

    #with open(Path(f"mappings/{protocol_id}.json")) as f:
    #    mapping_file = json.load(f)
    
    mapper = conversationMapper(mapping_file)

    t_delta = timedelta(days = 7) # get past 7 days of data
    query_date = (datetime.now() - t_delta)
    query = db_pro.collection("clients").document(client_id).collection("conversations").where("date", ">=", query_date).where("protocolId", "==", protocol_id).where("answered", "==", True).get()
    
    conversations = []
    for doc in query:
        conversations.append(mapper.map_conversation(conversation=doc.to_dict(), id=doc.id))
    df = pd.DataFrame(conversations)
    # There is a bug with INT64 in df.to_gbq() so we need to do this shit
    temp_csv_string = df.to_csv(sep=";", index=False)
    temp_csv_string_IO = StringIO(temp_csv_string)
    df = pd.read_csv(temp_csv_string_IO, sep=";")
    # This new df can be uploaded to BQ with no issues, go figure
    df.to_gbq(table_id, project_id, if_exists="append")

    return table_id,200