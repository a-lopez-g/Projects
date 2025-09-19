import copy
from pprint import pprint

def clean_intent(intent):
    return intent.replace(" ","").lower()
    
def get_intents_entities(conversation):
    try:
        intents = conversation['intentUpdated']
        entities = conversation['entityUpdated']
    except:
        intents = conversation['intent']
        entities = conversation['entity']
    entity_contents = get_entity_contents(intents, entities)

    return intents, entity_contents

def get_entity_contents(intents, entities):
    clean_entities = get_clean_entities(entities)
    entity_contents = []
    for i, entity in enumerate(clean_entities):
        if any(entity):
            contents = {}
            contents['origin_intent'] = intents[i]
            contents['values'] = [value for value in entity.values()]
            entity_contents.append(contents)

    return entity_contents

def get_clean_entities(entities):
    # TODO: this sometimes breaks.
    new_entities = copy.deepcopy(entities)
    for i, entity in enumerate(entities):
        try:
            for key, values in entity.items():
                try:
                    value = values["listValue"]["values"][0]
                except:
                    new_entities[i].pop(key)  # df returns entityType but empty value
                    continue
                if value["kind"] == "stringValue":
                    display_entity = str(value["stringValue"]).capitalize()
                else:  # elif value["kind"] == "structValue": composite entity
                    fields = value["structValue"]["fields"]
                    display_entity = ""
                    for field_name, field_value in sorted(fields.items()):
                        display_entity = display_entity + " " + str(field_value["stringValue"]).capitalize()
                new_entities[i][key.lower()] = display_entity
        except:
            print(entity)

    return new_entities

class ConversationToVariableMapper:
    """
    Functions required to map a conversation to the relevant protocol variables. 
    """
    def __init__(self, mapping_file=None):
        if mapping_file is None:
            print("Warning: Warning: mapping file not specified.")
            self.mapping_dict = {}
            self.variables = {}
            self.intent_mapping = {}
            self.entity_mapping = {}
        else:
            self.mapping_dict = mapping_file
            self.variables = mapping_file['variables']
            self.intent_mapping = mapping_file['intent_mapping']
            self.entity_mapping = mapping_file['entity_mapping']

    def display_mappings(self, mapping='full'):
        if mapping == 'full':
            pprint(self.mapping_dict)
        elif mapping == 'variables':
            print(self.variables)
        elif mapping == 'intents':
            pprint(self.intent_mapping)
        elif mapping == 'entities':
            pprint(self.entity_mapping)
        else:
            print(f"Mapping '{mapping}' not defined. Please select one of the following:")
            print("'full', 'variables', 'intents' or 'entities'.")

    def init_variables(self):
        """TO DO: This needs to be a bit more fleshed out."""
        return dict.fromkeys(self.variables, 0)
    
    def map_conversation(self, conversation, id, verbose=False):
        mapped_conversation = {}
        mapped_conversation['conversation_id'] = id
        mapped_conversation['patient_id'] = conversation['patientId']
        mapped_conversation['date'] = conversation['date']
        intents, entity_contents = get_intents_entities(conversation)
        mapped_variables = self.map_variables(intents, entity_contents, verbose)
        mapped_conversation.update(mapped_variables)

        return mapped_conversation

    def map_variables(self, intents, entity_contents, verbose=False) -> dict:
        new_variables = self.init_variables()
        for intent in intents:
            intent_maps = self.intent_mapping.get(clean_intent(intent))
            if intent_maps is not None:
                for var, value in intent_maps.items():
                    new_variables[var] = value
            else:
                if verbose:
                    print(f"Warning: intent '{intent}' not in intent mapping dictionary.")

        for contents in entity_contents:
            origin_intent = clean_intent(contents['origin_intent'])
            values = contents['values']
            map = self.entity_mapping.get(origin_intent)
            if map is not None:
                if map[1] == 'int':
                    try:
                        new_variables[map[0]] = int(values[-1])
                    except:
                        if verbose:
                            print(f"Failed to convert '{map[0]}'.")
                elif map[1] == 'float':
                    try:
                        new_variables[map[0]] = float(values[-1])
                    except:
                        if verbose:
                            print(f"Failed to convert '{map[0]}'.")
                elif map[1] == 'string':
                    new_variables[map[0]] = str.join(", ", values)
            
                elif map[1] == 'compound':
                    #print(values[0].split())
                    new_variables[map[0]] = values[0].split() 

                else:
                    print(f'Warning: unknown conversion rule: {map[1]}')
            else:
                if verbose:
                    print(f"Warning: origin intent '{origin_intent}' not in entity mapping dictionary.")

        return new_variables

    