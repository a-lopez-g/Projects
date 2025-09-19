import time

class Inference():
    def __init__(self):
        pass

    def execute(self, test_data, model, tokenizer):
        # Save inference results
        predictions = []
        prediction_times = []

        for conversation  in test_data: 
            prompt = conversation["text"]
            # Opción 1
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True,add_special_tokens=False).to("cuda")
            start_time = time.time()
            outputs = model.generate(**inputs, max_new_tokens=10,pad_token_id=tokenizer.eos_token_id) 
            end_time = time.time()
            inference_time = end_time - start_time
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Opción 2 -> Hay que aumentar el max_n_tokens más de 10 veces, tarda mucho más y es más difícil de controlar el formato de salida
            # text, inference_time = self.inference_using_encode_plus(model, tokenizer, prompt)

            predictions.append(text)
            prediction_times.append(inference_time)
            
        # TODO: review esto
        # test_data.update({
        #     "predictions": predictions,
        #     "prediction_times": prediction_times
        # })

        return predictions, prediction_times
        
    def compute_max_new_tokens():
        pass

    def inference_using_encode_plus(self, model, tokenizer, prompt):
        inputs = tokenizer.encode_plus(
            prompt,
            return_tensors='pt',
            max_length=100,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=False
            )

        inputs['input_ids'] = inputs['input_ids'].to(model.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(model.device)

        print("Inferring...")
        start_time = time.time()
        completion_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            num_return_sequences=1
        )
        end_time = time.time()
        inference_time = end_time - start_time
        completion = tokenizer.decode(completion_ids[0])
        return completion, inference_time

