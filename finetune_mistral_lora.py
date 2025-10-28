"""
Fine-tune Mistral model using LoRA on Alzheimer's dataset
This script implements efficient fine-tuning using Parameter-Efficient Fine-Tuning (PEFT)
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # You can also use "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "alzheimers_dataset.json"
OUTPUT_DIR = "./mistral-alzheimers-lora"
MAX_LENGTH = 512

# LoRA Configuration
LORA_R = 16  # Rank
LORA_ALPHA = 32  # Alpha scaling
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100
SAVE_STEPS = 50
LOGGING_STEPS = 10


def load_dataset(file_path):
    """
    Load and format the Alzheimer's dataset from JSON file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Format the data into training examples
    formatted_data = []
    for item in data:
        # Create instruction-following format
        text = f"""<s>[INST] {item['instruction']}

{item['input']} [/INST] {item['output']}</s>"""

        formatted_data.append({"text": text})

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(formatted_data)

    print(f"Loaded {len(dataset)} training examples")
    return dataset


def create_bnb_config():
    """
    Create BitsAndBytes configuration for 4-bit quantization
    This reduces memory usage significantly
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bnb_config


def create_lora_config():
    """
    Create LoRA configuration for parameter-efficient fine-tuning
    """
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        inference_mode=False
    )
    return lora_config


def load_model_and_tokenizer():
    """
    Load Mistral model with 4-bit quantization and tokenizer
    """
    print(f"Loading model: {MODEL_NAME}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    bnb_config = create_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Disable cache for training
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print("Model loaded successfully")
    return model, tokenizer


def prepare_model_for_training(model):
    """
    Prepare the model for training with LoRA
    """
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def create_training_arguments():
    """
    Create training arguments for the trainer
    """
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_steps=WARMUP_STEPS,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",  # Change to "tensorboard" or "wandb" if you want logging
        save_total_limit=3,
    )
    return training_args


def train_model():
    """
    Main training function
    """
    print("=" * 50)
    print("Starting Mistral Fine-tuning with LoRA")
    print("=" * 50)

    # Load dataset
    print("\n1. Loading dataset...")
    dataset = load_dataset(DATASET_PATH)

    # Load model and tokenizer
    print("\n2. Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # Prepare model for training
    print("\n3. Preparing model with LoRA...")
    model = prepare_model_for_training(model)

    # Create training arguments
    print("\n4. Setting up training configuration...")
    training_args = create_training_arguments()

    # Create trainer
    print("\n5. Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=create_lora_config(),
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # Start training
    print("\n6. Starting training...")
    print("=" * 50)
    trainer.train()

    # Save the final model
    print("\n7. Saving model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 50)


def test_model():
    """
    Test the fine-tuned model with a sample query
    """
    print("\n" + "=" * 50)
    print("Testing the fine-tuned model...")
    print("=" * 50)

    # Load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)

    # Test prompt
    test_prompt = """<s>[INST] Analyze the following patient symptoms and provide a risk assessment for Alzheimer's disease.

Patient: 68-year-old male, experiencing memory difficulties, confusion with familiar places, and changes in judgment. [/INST]"""

    # Generate response
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nTest Prompt:")
    print(test_prompt)
    print("\nModel Response:")
    print(response)
    print("=" * 50)


if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU detected. Training will be very slow on CPU.")
        print("Consider using Google Colab or a machine with GPU for efficient training.")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Train the model
    train_model()

    # Optionally test the model (uncomment if you want to test after training)
    # test_model()

    print("\n✓ Fine-tuning script completed successfully!")
    print(f"\nTo use your fine-tuned model:")
    print(f"1. Load the adapter: PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")
    print(f"2. Or merge and save: model.merge_and_unload()")
