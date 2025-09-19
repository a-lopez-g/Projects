import re

def format_input_data(few_shot,n_shots,datasets_path, datasets_description, datasets_target,datasets_type) -> dict :

    input_data = {
                "few_shot": few_shot,
                "n_shots": n_shots,
                # Array de databases
                "datasets_info": []
    }

    if len(datasets_path) != len(datasets_description) != len(datasets_target): 
        return "Something goes wrong. Review input data size!"
    
    for path,description,target,type in zip(datasets_path,datasets_description,datasets_target,datasets_type):
        input_data["datasets_info"].append(
            {
            "path": path,
            "description": description,
            "target_column":target,
            "database_type": type
            }
        )

    return input_data

def postprocess_groundtruths(targets: list, target_patterns: list) -> list:
    groundtruth = []
    for text in targets:
        for pattern in target_patterns:
            match = re.search(pattern, text)
            if match:
                found_class = match.group(1)
                groundtruth.append(found_class)
                break
        else:
            groundtruth.append(None)
    return groundtruth

def postprocess_predictions(predictions: str, patterns: list, groundtruths: list) -> str: 
    postprocessed_values = []
    text_prediction_to_fix = []
    for i,prediction in enumerate(predictions): 
        cleaned_prediction = re.sub(r'[^\w\s:\.\,\(\)\[\]\-\n]', '', prediction)
        for pattern, classes in patterns:
            match = re.search(pattern, cleaned_prediction)
            if match:
                found_class = match.group(1)
                if found_class in classes:
                    postprocessed_values.append(found_class)
                    break
        else:
            found_class = list(classes - {groundtruths[i]})[0]
            postprocessed_values.append(found_class)
            text_prediction_to_fix.append(i)
    return postprocessed_values,text_prediction_to_fix