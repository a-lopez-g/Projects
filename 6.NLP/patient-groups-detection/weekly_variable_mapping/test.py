from main import weekly_variable_mapping

GOOGLE_APPLICATION_CREDENTIALS = (
    "/home/andrea/Desktop/neotec/weekly_variable_mapping/credentials/test-key.json"
)


class Data:
    def __init__(self):
        self.data = {
            "client_id":"C-0006",  
            "protocol_id": "RSA-0002-Pteuz-P-IMjuq"
        }
        

    def get_json(self):
        return self.data


data = Data()

table_id, status = weekly_variable_mapping("C-0006","RSA-0002-Pteuz-P-IMjuq")
print(table_id, status)