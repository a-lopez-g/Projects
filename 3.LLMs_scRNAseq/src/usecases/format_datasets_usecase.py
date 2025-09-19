import pandas as pd
from datasets import concatenate_datasets, Dataset, Features, Value
from sklearn.model_selection import train_test_split
import random
import re

class FormatDataset:
    def __init__(self, test_size=0.2, validation_size=0.2, random_state=42):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    def execute(self, tokenizer, input_data: dict, prompt: str, completion: str):
        """
        Main function to execute dataset formatting based on the provided input data.
        """
        few_shot = input_data["few_shot"]
        n_shots = input_data["n_shots"]
        datasets_info = input_data["datasets_info"]

        train_dataset, val_dataset, test_dataset = self.initialize_datasets()

        for dataset in datasets_info:
            dataset_type = dataset.get("database_type")
            dataset_path = dataset["path"]
            target_column = dataset["target_column"]
            description = dataset["description"]

            train_df, val_df, test_df = self.load_and_split_dataset(dataset_path, target_column, dataset_type)

            train_dataset, val_dataset, test_dataset = self.process_datasets(
                train_df, val_df, test_df, tokenizer, target_column, prompt, completion, 
                description, few_shot, n_shots, train_dataset, val_dataset, test_dataset
            )

        print(f"Number of training samples: {train_dataset.num_rows}")
        print(f"Number of validation samples: {val_dataset.num_rows}")
        print(f"Number of testing samples: {test_dataset.num_rows}")
        return train_dataset, val_dataset, test_dataset

    def initialize_datasets(self):
        """
        Initialize empty datasets for train, validation, and test.
        """
        empty_dataset = Dataset.from_dict({"text": []}).cast(Features({'text': Value('string')}))
        return empty_dataset, empty_dataset, empty_dataset

    def load_and_split_dataset(self, dataset_path, target_column, dataset_type):
        """
        Load and split the dataset based on its type.
        """
        if dataset_type == "train":
            data = pd.read_csv(dataset_path)
            train_df, val_df = train_test_split(
                data, test_size=self.test_size, stratify=data[target_column], random_state=self.random_state
            )
            return train_df, val_df, None
        elif dataset_type == "test":
            test_df = pd.read_csv(dataset_path)
            return None, None, test_df
        else:
            train_df, val_df, test_df = self.create_df_from_database(dataset_path, target_column)
            return train_df, val_df, test_df

    def process_datasets(self, train_df, val_df, test_df, tokenizer, target_column, prompt, completion, 
                         description, few_shot, n_shots, train_dataset, val_dataset, test_dataset):
        """
        Process train, validation, and test datasets by creating conversations and combining them.
        """
        few_shot_conversations=[]
        
        if train_df is not None:
            train_conversations = self.create_conversations(train_df, target_column, prompt, completion, description)
            val_conversations = self.create_conversations(val_df, target_column, prompt, completion, description)
            train_dataset = self.combine_datasets(train_dataset, train_conversations, tokenizer, "train")
            val_dataset = self.combine_datasets(val_dataset, val_conversations, tokenizer, "train")

        if test_df is not None:
            test_conversations, completions = self.create_conversations(test_df, target_column, prompt, completion, description, is_test=True)
            if few_shot:
                few_shot_conversations, test_conversations, completions = self.get_few_shot_conversations(
                    n_shots, test_conversations, completions
                )
            test_dataset = self.combine_datasets(
                test_dataset, test_conversations, tokenizer, "test", completions, few_shot_conversations
            )

        return train_dataset, val_dataset, test_dataset

    def create_df_from_database(self, database_path, target_column):
        """
        Load the dataset and split it into train, validation, and test sets.
        """
        data = pd.read_csv(database_path)
        train_data, test_data = train_test_split(
            data, test_size=self.test_size, stratify=data[target_column], random_state=self.random_state
        )
        train_data, val_data = train_test_split(
            train_data, test_size=self.validation_size, stratify=train_data[target_column], random_state=self.random_state
        )
        return train_data, val_data, test_data

    def create_conversations(self, df, target_column, prompt, completion, description="", is_test=False):
        """
        Create conversations and completions based on the dataset.
        """
        conversations, completions = [], []
        possible_values = sorted(df[target_column].unique())
        feature_columns = [col for col in df.columns if col != target_column]

        for _, row in df.iterrows():
            features_text = ", ".join([f"{col} is {row[col]}" for col in feature_columns])
            formatted_prompt = self.format_text(prompt, target_column, possible_values, features_text, row[target_column])
            formatted_completion = self.format_text(completion, target_column, possible_values, features_text, row[target_column])

            conversation = [
                {'role': 'system', 'content': f"I am a model predicting class values. {description}"},
                {'role': 'user', 'content': formatted_prompt}
            ]
            if not is_test:
                conversation.append({'role': 'assistant', 'content': formatted_completion})
            conversations.append(conversation)
            completions.append(formatted_completion)

        return (conversations, completions) if is_test else conversations

    def format_text(self, template, target_column, possible_values, features_text, output_value):
        """
        Replace placeholders in the text with actual values.
        """
        def format_value(value):
             # If it's a number and ends with .0, convert it to integer
            if isinstance(value, (float, int)) and float(value).is_integer():
                 return int(value)
            return value

        replacements = {
            "'target_column'": target_column,
            "'classes'": f"{[format_value(v) for v in possible_values]}",
            "'classes[0]'": f"{format_value(possible_values[0])}",
            "'features_text'": features_text,
            "'output'": f"{format_value(output_value)}"
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        return template

    def combine_datasets(self, combined_dataset, conversations, tokenizer, dataset_type, targets=None, few_shot_conversations=[]):
        """
        Combine the current dataset with new conversations and return the updated dataset.
        """
        dataset = self.create_dataset(conversations, tokenizer, dataset_type, targets, few_shot_conversations)
        return concatenate_datasets([combined_dataset, dataset]).shuffle(seed=self.random_state)

    def create_dataset(self, conversations, tokenizer, dataset_type, targets=None, few_shot_conversations=[]):
        """
        Create a HuggingFace Dataset from conversations.
        """
        prompts = []
        for conversation in conversations:
            fs_conversations = few_shot_conversations.copy()
            if few_shot_conversations: 
                fs_conversations.append(conversation)
                flattened_conversations = [item for sublist in fs_conversations for item in sublist]
                prompt = tokenizer.apply_chat_template(
                    flattened_conversations,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else: 
                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=(dataset_type == "test")
                )
            prompts.append(prompt)

        dataset_dict = {"text": prompts}
        if dataset_type == "test":
            dataset_dict["targets"] = targets
        return Dataset.from_dict(dataset_dict)

    def get_few_shot_conversations(self, n_shots, test_conversations, completions):
        """
        Select a few-shot subset of the test conversations for testing purposes.
        """
        indices = random.sample(range(len(test_conversations)), min(n_shots, len(test_conversations)))
        few_shot_conversations = [test_conversations[i] + [{'role': 'assistant', 'content': completions[i]}] for i in indices]
        test_conversations = [conv for i, conv in enumerate(test_conversations) if i not in indices]
        completions = [comp for i, comp in enumerate(completions) if i not in indices]
        return few_shot_conversations, test_conversations, completions



# class FormatDataset(): 
#     def __init__(self) -> None:
#         self.test_size = 0.2
#         self.validation_size = 0.2
#         self.random_state = 42

#     def execute(self, tokenizer, input_data: dict, prompt: str, completion: str): 

#         few_shot = input_data["few_shot"]
#         n_shots = input_data["n_shots"]
#         datasets_info = input_data["datasets_info"]

#         train_combined_dataset = Dataset.from_dict({"text": []})
#         validation_combined_dataset = Dataset.from_dict({"text": []})
#         test_combined_dataset = Dataset.from_dict({"text": []})

#         for dataset in datasets_info: 
#             dataset_path = dataset["path"]
#             dataset_description = dataset["description"]
#             dataset_type = dataset["database_type"]
#             target_column = dataset["target_column"]
#             few_shot_conversations = []

#             # Option 1. USE ALL DATABASES AS TRAIN+VALIDATION+TEST
#             if not dataset_type:
#                 # Load & split database, create dataframe
#                 train_df, validation_df, test_df = self.create_df_from_database(dataset_path, target_column)

#                 # Create training, validation and testing conversations 
#                 train_conversations = self.create_conversations_from_df(train_df, target_column, prompt, completion, description=dataset_description,is_test=False)
#                 validation_conversations = self.create_conversations_from_df(validation_df, target_column, prompt, completion, description=dataset_description, is_test=False)
#                 test_conversations, test_completions = self.create_conversations_from_df(test_df, target_column, prompt, completion, description=dataset_description, is_test=True)
                
#                 # Few shot conversations
#                 if few_shot: 
#                     few_shot_conversations = self.get_few_shot_conversations(few_shot,n_shots,validation_conversations)

#                 # Create training, validation and testing datasets
#                 train_dataset = self.create_dataset_from_conversations(train_conversations,tokenizer,dataset_type="train") 
#                 validation_dataset = self.create_dataset_from_conversations(validation_conversations, tokenizer,dataset_type="train")
#                 test_dataset = self.create_dataset_from_conversations(test_conversations, tokenizer,few_shot_conversations,dataset_type="test",targets=test_completions,add_generation_prompt=True)
#                 # Mix datasets
#                 train_combined_dataset = (concatenate_datasets([train_combined_dataset, train_dataset])).shuffle(seed=42) 
#                 validation_combined_dataset = (concatenate_datasets([validation_combined_dataset, validation_dataset])).shuffle(seed=42) 
#                 test_combined_dataset = (concatenate_datasets([test_combined_dataset, test_dataset])).shuffle(seed=42) 

#             # Option 2. USE DATABASE AS TRAINING SET
#             elif dataset_type=="train":
#                 # Create dataframe & Split
#                 training_df = pd.read_csv(dataset_path)
#                 stratify = training_df[target_column]
#                 train_df, validation_df = train_test_split(training_df,test_size=self.test_size,random_state=self.random_state,stratify=stratify)

#                 # Create training and validation conversations 
#                 train_conversations = self.create_conversations_from_df(train_df, target_column, prompt, completion, description=dataset_description,is_test=False)
#                 validation_conversations = self.create_conversations_from_df(validation_df, target_column, prompt, completion, description=dataset_description, is_test=False)
                
#                 # Create training and validation datasets 
#                 train_dataset = self.create_dataset_from_conversations(train_conversations, tokenizer,dataset_type="train") 
#                 validation_dataset = self.create_dataset_from_conversations(validation_conversations, tokenizer, dataset_type="train")
                
#                 # Mix datasets
#                 train_combined_dataset = (concatenate_datasets([train_combined_dataset, train_dataset])).shuffle(seed=42) 
#                 validation_combined_dataset = (concatenate_datasets([validation_combined_dataset, validation_dataset])).shuffle(seed=42) 
                
#             # Option 3. USE DATABASE AS TEST
#             else:
#                 # Create dataframe
#                 testing_df = pd.read_csv(dataset_path)
#                 # Create test conversations
#                 test_conversations, test_completions = self.create_conversations_from_df(testing_df, target_column, prompt, completion, description=dataset_description, is_test=True)
                
#                 # Few-shot conversations
#                 if few_shot: 
#                     few_shot_conversations,test_conversations,test_completions = self.get_few_shot_conversations(few_shot,n_shots,validation_conversations=test_conversations,dataset_type=dataset_type,test_conversations=test_conversations,test_completions=test_completions)

#                 # Create test dataset
#                 test_dataset = self.create_dataset_from_conversations(test_conversations, tokenizer, few_shot_conversations, dataset_type="test", targets=test_completions, add_generation_prompt=True)
#                 # Mix datasets
#                 test_combined_dataset = (concatenate_datasets([test_combined_dataset, test_dataset])).shuffle(seed=42) 
             

#         print(f"Number of training samples: {train_combined_dataset.num_rows}")
#         print(f"Number of validation samples: {validation_combined_dataset.num_rows}")
#         print(f"Number of testing samples: {test_combined_dataset.num_rows}")


#         return train_combined_dataset, validation_combined_dataset, test_combined_dataset   


#     def create_df_from_database(self, database_path, target_column):
#         # Load the dataset
#         data_df = pd.read_csv(database_path)
        
#         # Split the data into training, validation and testing sets
#         stratify = data_df[target_column]

#         training_df, test_df = train_test_split(
#             data_df,
#             test_size=self.test_size,
#             random_state=self.random_state,
#             stratify=stratify
#             )
        
#         stratify = training_df[target_column]
#         train_df, validation_df = train_test_split(
#             training_df,
#             test_size=self.validation_size, 
#             random_state=self.random_state,
#             stratify=stratify
#         )
#         return train_df, validation_df, test_df

#     def create_conversations_from_df(self, df, target_column, input_prompt, input_completion, description="", is_test=False):
#         conversations = []  # List to store all conversations
#         completions = []  # List to store all completions

#         possible_values = sorted(df[target_column].unique())
#         feature_columns = [col for col in df.columns if col != target_column]
        
#         def format_value(value):
#             # If it's a number and ends with .0, convert it to integer
#             if isinstance(value, (float, int)) and float(value).is_integer():
#                 return int(value)
#             return value
        
#         # Función para realizar sustituciones automáticas en el prompt
#         def replace_strings(text, replacements):
#             new_text = text  # Copia explícita del texto
#             for old, new in replacements.items():
#                 new_text = new_text.replace(old, new)
#             return new_text
        
#         for index, row in df.iterrows():
#             conversation = []  # List to store messages for one conversation
            
#             # Format each feature using format_value function
#             features_text = ", ".join([
#                 f"{col} is {format_value(row[col])}" 
#                 for col in feature_columns
#             ])

#             # Format prompt
#             replacements = {
#                         "'target_column'": f"{target_column}",
#                         "'classes'": f"{[format_value(v) for v in possible_values]}",
#                         "'classes[0]'": f"{[format_value(v) for v in possible_values][0]}",
#                         "'features_text'": f"{features_text}",
#                         "'output'": f"{format_value(row[target_column])}"
#                     }
            
#             # Create complete prompt including possible values
#             prompt = replace_strings(input_prompt, replacements)
#             # Get the formatted target value for this row
#             completion = replace_strings(input_completion, replacements)

#             # Append system message to conversation
#             conversation.append({
#                 'role': 'system',
#                 'content': 'I am a model that can predict the class value of a feature based on the values of other features in this dataset. ' + description

#             })
            
#             # Append user message to conversation
#             conversation.append({
#                 'role': 'user',
#                 'content': prompt
#             })
            
#             # Append assistant message to conversation only if it's not a test
#             if not is_test:
#                 conversation.append({
#                     'role': 'assistant',
#                     'content': completion
#                 })
            
#             # Add this conversation to the list of conversations
#             conversations.append(conversation)

#             # Add the completion to the list of completions
#             completions.append(completion)

#         if not is_test:
#             return conversations
#         else:
#             return conversations, completions

#     def create_dataset_from_conversations(
#             self,
#             conversations, 
#             tokenizer,
#             few_shot_conversations=[],
#             dataset_type="train",
#             targets=None,
#             add_generation_prompt=False):

#         # Validate inputs
#         if dataset_type not in ["train", "test"]:
#             raise ValueError("dataset_type must be either 'train' or 'test'")
            
#         if dataset_type == "test" and targets is None:
#             raise ValueError("targets must be provided for test dataset")
        
#         # Create a list to store all processed prompts
#         prompts = []
        
#         # Process each conversation
#         for conversation in conversations:
#             fs_conversations = few_shot_conversations.copy()
#             if few_shot_conversations: 
#                 # Add the test conversation test/training
#                 fs_conversations.append(conversation)
#                 # Flatten all conversations including the test conversation
#                 flattened_conversations = [item for sublist in fs_conversations for item in sublist]
#                 # Apply the chat template
#                 prompt = tokenizer.apply_chat_template(
#                     flattened_conversations,
#                     tokenize=False,
#                     add_generation_prompt=True
#                 )

#             else: 
#                 prompt = tokenizer.apply_chat_template(
#                     conversation,
#                     tokenize=False,
#                     add_generation_prompt=add_generation_prompt
#                 )
#             prompts.append(prompt)
        
#         # Create dataset dictionary based on type
#         if dataset_type == "train":
#             dataset_dict = {
#                 "text": prompts
#             }
#         else:  # test dataset
#             dataset_dict = {
#                 "text": prompts,
#                 "targets": targets
#             }
        
#         # Create and return the dataset
#         return Dataset.from_dict(dataset_dict)

#     def get_few_shot_conversations(self,few_shot, n_shots, validation_conversations,dataset_type="",test_conversations=[],test_completions=[]): 
#         # Ensure num_examples is not larger than available conversations
#         num_examples = min(n_shots, len(validation_conversations)) 
#         # Randomly select conversations from validation set
#         selected_indices = random.sample(range(len(validation_conversations)), num_examples)
#         # If we get few conversations from test we need to delete it in test_conversations list
#         if dataset_type == "test": 
#             few_shot_conversations = []
#             for i in selected_indices:
#                 conver = validation_conversations[i]
#                 completion = test_completions[i]
#                 conver.append({
#                 'role': 'assistant',
#                 'content': completion
#                 })
#                 few_shot_conversations.append(conver)
#             test_conversations = [val for idx, val in enumerate(test_conversations) if idx not in selected_indices]
#             test_completions = [val for idx, val in enumerate(test_completions) if idx not in selected_indices]
#             return few_shot_conversations,test_conversations,test_completions
#         else:
#             few_shot_conversations=[validation_conversations[i] for i in selected_indices]
#             return few_shot_conversations