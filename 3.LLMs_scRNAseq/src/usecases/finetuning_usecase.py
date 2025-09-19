from transformers import TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig
import bitsandbytes as bnb
from trl import SFTTrainer
import wandb

class FinetuningLlamaModel():
    def __init__(self):
        # TODO: añadir como parámetros de entrada
        self.r = 64
        self.lora_alpha = 16
        self.lora_dropout = 0

    def execute(self, model, tokenizer, finetuned_model_name, train_dataset, eval_dataset): 

        # TODO: poner bien
        # self.steps_per_epoch = len(train_dataset) // 1  # Per device batch size of 1
        # self.max_steps = self.steps_per_epoch * 50  # Force exactly 50 epochs
        # print(self.max_steps)
        
        # Setting up the model
        print("Settiung up model...")
        # Extract the linear model name from the model.
        modules = self.find_all_linear_names(model)
        peft_config = self.setup_Lora_config(modules)
        training_arguments = self.setup_training_arguments(finetuned_model_name)
        # Setting sft parameters
        trainer = self.setupt_SFT_parameters(
            base_model=model,
            tokenizer=tokenizer,
            peft_config=peft_config,
            training_arguments=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Añado EarlyStopping
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

        # Model training
        print("Training model...")
        #checkpoint = "./Llama-3.1-8B-NHANES-DIABETES130US/checkpoint-1900"
        #resume_from_checkpoint=checkpoint
        trainer.train()
        # wandb.finish()
        # Evaluate model
        results = trainer.evaluate()
        print("Resultados de evaluación:", results)

        # Save the fine-tuned model
        print("Saving model...")
        # trainer.model.save_pretrained(new_model)
        trainer.model.push_to_hub(finetuned_model_name, use_temp_dir=False)
        # TODO: no hace falta?
        tokenizer.push_to_hub(finetuned_model_name, use_temp_dir=False)

        return trainer.model, tokenizer


    def find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16 bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def setup_Lora_config(self, modules): 
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules
        )
    
    def setup_training_arguments(self, finetuned_model_name):
        return TrainingArguments(
                                    output_dir=finetuned_model_name,                    # directory to save and repository id
                                    num_train_epochs=50,                       # number of training epochs
                                    per_device_train_batch_size=1,            # batch size per device during training
                                    gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
                                    gradient_checkpointing=True,              # use gradient checkpointing to save memory
                                    evaluation_strategy="steps",             # evalúa en intervalos de steps
                                    eval_steps=50,                           # realiza evaluación cada 50 pasos
                                    save_strategy="steps",
                                    save_steps=50,
                                    save_total_limit=3,                      # guarda un máximo de 3 checkpoint
                                    optim="paged_adamw_32bit",
                                    logging_steps=1,                         
                                    learning_rate=2e-4,                       # learning rate, based on QLoRA paper
                                    weight_decay=0.001,
                                    fp16=False,
                                    bf16=False,
                                    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
                                    max_steps=-1,                       
                                    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
                                    group_by_length=True,
                                    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
                                    # report_to="wandb",                  # report metrics to w&b
                                    load_best_model_at_end=True,             # carga el mejor modelo al final
                                    metric_for_best_model="eval_loss",       # usa 'eval_loss' como métrica
                                    greater_is_better=False,                # menor eval_loss es mejor
                                )

    def setupt_SFT_parameters(self, base_model, tokenizer, peft_config, training_arguments, train_dataset, eval_dataset): 
        return SFTTrainer(
                            model=base_model,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            peft_config=peft_config,
                            max_seq_length= 512,
                            dataset_text_field="text",
                            tokenizer=tokenizer,
                            args=training_arguments,
                            packing= False,
                            dataset_kwargs=
                            {
                            "add_special_tokens": False, # If you format text with apply_chat_template(tokenize=False), you should set the argument add_special_tokens=False when you tokenize that text later. If you use apply_chat_template(tokenize=True), you don’t need to worry about this!
                            "append_concat_token": False,
                            }
                        )