from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
    )
from peft import PeftModel
from trl import setup_chat_format
import torch

# CHAT_TEMPLATE = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
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


def load_model_and_tokenizer(model_name: str,chat_templates=CHAT_TEMPLATE): 
    # TODO: review esto
    device_map = torch.device("cuda:0") 
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True) 
    # Load the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Load the model in 4-bit precision
        bnb_4bit_use_double_quant=False,  # Do not use double quantization
        bnb_4bit_quant_type="nf4",  # Set the quantization type to nf4 
        bnb_4bit_compute_dtype="float16",  # Use float16 for computation # If we use flash -> "bfloat16"
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # Name or path of the pre-trained model to load
        return_dict=True,  # Return model outputs as a dictionary instead of tuples
        low_cpu_mem_usage=True,  # Optimize RAM usage during model loading
        torch_dtype=torch.float16,  # Set model parameters to 16-bit precision to reduce memory usage and increase speed
        device_map=device_map,  # Automatically distribute model layers across available hardware (CPU/GPU/TPU)
        trust_remote_code=True,  # Allow execution of model-specific code (use with trusted sources only)
        quantization_config=bnb_config,  # Use the custom BitsAndBytes configuration
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the end-of-sequence token
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Set the pad token id to the end-of-sequence token id
    model.config.pad_token_id = model.config.eos_token_id  # Set the pad token id to the end-of-sequence token id in model
    
     # Chat template of the tokenizer (based on Llama's chat template)
    tokenizer.chat_template = chat_templates
    return model, tokenizer

def load_finetuned_model(base_model_reload, finetuned_model): 
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model)
    # TODO: hace falta? -> NO.
    # base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
    model = PeftModel.from_pretrained(base_model_reload, finetuned_model)
    model = model.merge_and_unload()
    return model, tokenizer


