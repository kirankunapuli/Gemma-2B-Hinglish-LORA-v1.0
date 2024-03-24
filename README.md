# 🔥 [Gemma-2B-Hinglish-LORA-v1.0 model](https://huggingface.co/kirankunapuli/Gemma-2B-Hinglish-LORA-v1.0)
## 🚀 Visit this HF 🤗 Space to try out this model's inference: https://huggingface.co/spaces/kirankunapuli/Gemma-2B-Hinglish-Model-Inference-v1.0

- **Developed by:** [Kiran Kunapuli](https://www.linkedin.com/in/kirankunapuli/)
- **License:** apache-2.0
- **Finetuned from model :** unsloth/gemma-2b-bnb-4bit
- **Model usage:** Use the below code in Python
  ```python
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained("kirankunapuli/Gemma-2B-Hinglish-LORA-v1.0")
    model = AutoModelForCausalLM.from_pretrained("kirankunapuli/Gemma-2B-Hinglish-LORA-v1.0")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {}
    
    ### Input:
    {}
    
    ### Response:
    {}"""
  
    # Example 1
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Please answer the following sentence as requested", # instruction
            "ऐतिहासिक स्मारक India Gate कहाँ स्थित है?", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    output = tokenizer.batch_decode(outputs)[0]
    response_start = output.find("### Response:") + len("### Response:")
    response_end = output.find("<eos>", response_start)
    response = output[response_start:response_end].strip()
    print(response)
    
    # Example 2
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Please answer the following sentence as requested", # instruction
            "ऐतिहासिक स्मारक इंडिया गेट कहाँ स्थित है? मुझे अंग्रेजी में बताओ", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to(device)
    
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    output = tokenizer.batch_decode(outputs)[0]
    response_pattern = re.compile(r'### Response:\n(.*?)<eos>', re.DOTALL)
    response_match = response_pattern.search(output)
    if response_match:
        response = response_match.group(1).strip()
        return response
    else:
        return "Response not found"
  ```
- **Model config:**
  ```python
    model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, 
    bias = "none",   
    use_gradient_checkpointing = True, 
    random_state = 42,
    use_rslora = True,  
    loftq_config = None, 
    )
  ```
- **Training parameters:**
  ```python
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 120,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "wandb",
      ),
    )
  ```
- **Training details:**
  ```
  ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
     \\   /|    Num examples = 14,343 | Num Epochs = 1
  O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 4
  \        /    Total batch size = 8 | Total steps = 120
   "-____-"     Number of trainable parameters = 19,611,648
  GPU = Tesla T4. Max memory = 14.748 GB.
  2118.7553 seconds used for training.
  35.31 minutes used for training.
  Peak reserved memory = 9.172 GB.
  Peak reserved memory for training = 6.758 GB.
  Peak reserved memory % of max memory = 62.191 %.
  Peak reserved memory for training % of max memory = 45.823 %.
  ```

This gemma model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
