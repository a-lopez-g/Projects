import json

from src.usecases.format_datasets_usecase import FormatDataset
from src.usecases.finetuning_usecase import FinetuningLlamaModel
from src.usecases.inference_usecase import Inference

from src.utils.llm_utils import load_model_and_tokenizer,load_finetuned_model
from src.utils.dataset_utils import format_input_data,postprocess_groundtruths,postprocess_predictions
from src.utils.metrics import compute_prediction_metrics, graph_times

formatter = FormatDataset()
finetuning = FinetuningLlamaModel()
inference = Inference()

CHAT_TEMPLATE =  """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{# Extrae el mensaje del sistema para colocarlo en la sección correspondiente #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}

{{- system_message }}{{- "<|eot_id|>" }}

{# Procesa el primer mensaje del usuario y las herramientas si están habilitadas #}
{%- if tools_in_user_message and not tools is none %}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
    
    {{ '<|start_header_id|>user<|end_header_id|>\n\n' }}
    {{ "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n" }}
    {{ 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.\n\n' }}
    
    {%- for t in tools %}
        {{ t | tojson(indent=4) }}
        {{ "\n\n" }}
    {%- endfor %}
    
    {{ first_user_message + "<|eot_id|>" }}
{%- endif %}

{# Procesa el resto de los mensajes (user, assistant, tool calls, etc.) #}
{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content']|trim + '<|eot_id|>' }}
    
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
            {{ "<|python_tag|>" + tool_call.name + ".call(" }}
            
            {%- for arg_name, arg_val in tool_call.arguments.items() %}
                {{ arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}, {% endif %}
            {%- endfor %}
            
            {{ ")" }}
        {%- else %}
            {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
            {{ '{"name": "' + tool_call.name + '", "parameters": ' + tool_call.arguments | tojson + "}" }}
        {%- endif %}
        
        {{ "<|eot_id|>" }}
    
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{ "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {{ message.content | tojson if message.content is mapping else message.content }}
        {{ "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
    {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


## 1.Load model & tokenizer
# model_name = f"andrealopez/Llama-3.1-8B-NHANES-DIABETES130US"
model_name = "meta-llama/Llama-3.2-1B" #"meta-llama/Meta-Llama-3.2-1B"  # Name o path of the pre-trained model to load
model, tokenizer = load_model_and_tokenizer(model_name=model_name)

## 2.Format database to prompts
databases = {
    # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database -> Binarypima_path = './datasets/PIMA_diabetes.csv'
    "PIMA": {
        "path": "./datasets/kaggle/PIMA_diabetes.csv",
        "description": "This dataset contains information about patients with diabetes.",
        "target_column": "Outcome",
        "type": ""
    },
    # https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset -> Binary
    "NHANES": {
        "path": "./datasets/NHANES_age_prediction.csv",
        "description": "This dataset contains information about the age of patients.",
        "target_column": "age_group",
        "type": "train"
    },
    # https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 ->"diabetesMed": Binary, "readmitted": Multiclass
    "Diabetes130US": {
        "path": "./datasets/diabetic_130US.csv",
        "description": "This dataset contains information about patients with diabetes.",
        "target_column": "readmitted",
        "type": "train"
    },
    # https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data -> Binary
    "BreastCancer": {
        "path": "./datasets/breast_cancer_diagnosis.csv",
        "description": "This dataset contains information about breast cancer diagnosis in patients.",
        "target_column": "diagnosis",
        "type": "train"
    },
    # https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset -> Binary
    "Alzheimer": {
        "path": "./datasets/alzheimers_diagnosis.csv",
        "description": "This dataset contains information about Alzheimer diagnosis in patients.",
        "target_column": "Diagnosis",
        "type": "train"
    },
    # https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset -> Multiclass
    "Stroke": {
        "path": "./datasets/stroke_diagnosis.csv",
        "description": "This dataset contains information about stroke diagnosis in patients.",
        "target_column": "stroke",
        "type": "train"
    },
    # https://www.kaggle.com/datasets/s3programmer/parkison-diseases-eeg-dataset -> Binary
    "Parkinson": {
        "path": "./datasets/parkinson_EEG_diagnosis.csv",
        "description": "This dataset contains information about Parkinson diagnosis based on EEG in patients.",
        "target_column": "class",
        "type": "train"
    },
    # https://www.kaggle.com/datasets/brunogrisci/leukemia-gene-expression-cumida -> Multiclass
    "Leukemia": {
        "path": "./datasets/leukemia_expression.csv",
        "description": "This dataset contains information about Leukemia diagnosis based on expression genes in patients.",
        "target_column": "type",
        "type": "train"
    },
    # https://www.kaggle.com/datasets/ruslankl/mice-protein-expression -> Multiclass
    "MiceTrisomic": {
        "path": "./datasets/mice_trisomic_expression.csv",
        "description": "This dataset contains information about protein expression in trisomic and health mice.",
        "target_column": "class",
        "type": "train"
    },
    # https://www.kaggle.com/datasets/ayamoheddine/pcos-dataset -> Binary
    "PCOS": {
        "path": "./datasets/PCOS.csv",
        "description": "This dataset contains information about POCS diagnosis in patients.",
        "target_column": "PCOS (Y/N)",
        "type": "train"
    }
}

syntehtic_datasets = {
    "PIMA": {
        "path": "./datasets/kaggle/PIMA_diabetes.csv",
        "description": "This dataset contains information about patients with diabetes.",
        "target_column": "Outcome",
        "type": "test"
    },
    "healthStatus":
    {
        "path": "./datasets/sinteticos/biological_data.csv",
        "description": "This dataset contains information about various biological and physiological parameters of different species.",
        "target_column": "health_status",
        "type": "train"
    },
    "readingBooks":
    {
        "path": "./datasets/sinteticos/book_reading_habits.csv",
        "description": "This dataset contains information about individuals' book reading habits.",
        "target_column": "reading_frequency",
        "type": "train"
    },
    "customerChurn":
    {
        "path": "./datasets/sinteticos/customer_churn.csv",
        "description": "This dataset contains information about customer churn.",
        "target_column": "churn",
        "type": "train"
    },
    "enviromentalRisk":
    {
        "path": "./datasets/sinteticos/environmental_monitoring.csv",
        "description": "This dataset contains information about environmental monitoring.",
        "target_column": "environmental_risk_level",
        "type": "train"
    },
    "wellness":
    {
        "path": "./datasets/sinteticos/health_wellness.csv",
        "description": "This dataset contains information about health and wellness metrics.",
        "target_column": "overall_wellness_level",
        "type": "train"
    },
    "streamingBehavior":
    {
        "path": "./datasets/sinteticos/movie_streaming_user_behavior.csv",
        "description": "This dataset contains information about user behavior on movie streaming platforms.",
        "target_column": "churn",
        "type": "train"
    },
    "shoppingBehavior":
    {
        "path": "./datasets/sinteticos/online_shopping_behavior.csv",
        "description": "This dataset contains information about online shopping behavior.",
        "target_column": "customer_loyalty_score",
        "type": "train"
    },
    "petAdoption":
    {
        "path": "./datasets/sinteticos/pet_adoption.csv",
        "description": "This dataset contains information about pet adoption, including details about the pets available for adoption.",
        "target_column": "adoption_speed",
        "type": "train"
    },
    "smartHome":
    {
        "path": "./datasets/sinteticos/smart_home_device_usage.csv",
        "description": "This dataset contains information about smart home device usage.",
        "target_column": "device_efficiency_level",
        "type": "train"
    },
    "urbanTraffic":
    {
        "path": "./datasets/sinteticos/urban_traffic_analysis.csv",
        "description": "This dataset contains information about urban traffic patterns.",
        "target_column": "traffic_congestion_level",
        "type": "train"
    }
}

# Config 
few_shot = False
n_examples_fewshot = 1

# "Diabetes130US"
# selected_databases = ["PIMA","NHANES","BreastCancer","Alzheimer","Stroke","Leukemia","Parkinson","MiceTrisomic","PCOS"]
# selected_databases = ["PIMA","healthStatus","readingBooks","customerChurn","enviromentalRisk","wellness","streamingBehavior","shoppingBehavior","petAdoption","smartHome","urbanTraffic"]  # Cambia los nombres según las bases de datos deseadas
selected_databases = ["PIMA"]
datasets_path = [databases[db]["path"] for db in selected_databases]
datasets_description = [databases[db]["description"] for db in selected_databases]
datasets_target = [databases[db]["target_column"] for db in selected_databases]
datasets_type = [databases[db]["type"] for db in selected_databases]

input_data = format_input_data(
    few_shot=few_shot,
    n_shots=n_examples_fewshot,
    datasets_path=datasets_path, 
    datasets_description=datasets_description, 
    datasets_target=datasets_target,
    datasets_type=datasets_type
)

# Define prompts
prompt = f"Predict the class value of 'target_column' among this classes: ('classes'), based on these features: 'features_text'. Instruction: Return only the predicted class using the following format: 'Predicted class: 'classes[0]''. What is the class value of 'target_column'?"
completion = f"'target_column': 'output'"

train_dataset, validation_dataset, test_dataset = formatter.execute(tokenizer,input_data,prompt,completion)

#train_dataset.to_csv("./datasets/train_all_datasets.csv")
#validation_dataset.to_csv("./datasets/validation_all_datasets.csv")
#test_dataset.to_csv("./datasets/test_all_datasets.csv") 

## 3.Finetuning
finetuned_model_name = "Llama-3.2-1B-Synthetic-PIMA"
finetuned_model, tokenizer = finetuning.execute(
    model=model,
    tokenizer=tokenizer, 
    finetuned_model_name=finetuned_model_name,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset
)

try: 
    predictions, prediction_times = inference.execute(
        test_data=test_dataset,
        model=finetuned_model,
        tokenizer=tokenizer
        )
except:
    # Cargar finetuned model desde HuggingFace
    finetuned_model_name = "Llama-3.2-1B-Synthetic-PIMA" 
    finetuned_model_path = f"andrealopez/{finetuned_model_name}"
    finetuned_model, tokenizer = load_model_and_tokenizer(model_name=finetuned_model_path)

    predictions, prediction_times = inference.execute(
        test_data=test_dataset,
        model=finetuned_model,
        tokenizer=tokenizer
        )
    

with open("./datasets/predictionsTest_syntheticPIMA.json", "w") as json_file:
    json.dump(predictions, json_file)

with open("./datasets/predictionsTimesTest_syntheticPIMA.json", "w") as json_file:
    json.dump(prediction_times, json_file)

