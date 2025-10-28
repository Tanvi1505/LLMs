# Mistral Fine-tuning with LoRA for Alzheimer's Dataset

This project provides a complete setup for fine-tuning the Mistral-7B model using LoRA (Low-Rank Adaptation) on an Alzheimer's disease dataset.

## Files

- `finetune_mistral_lora.py` - Main fine-tuning script
- `alzheimers_dataset.json` - Example dataset with 6 Alzheimer's-related samples
- `requirements.txt` - Python dependencies
- `README_FINETUNE.md` - This file

## Features

- **4-bit Quantization**: Uses BitsAndBytes for memory-efficient training
- **LoRA Adaptation**: Parameter-efficient fine-tuning (only ~0.1% of parameters trained)
- **Mistral-7B**: State-of-the-art open-source LLM
- **Instruction Format**: Properly formatted for instruction-following tasks
- **Ready to Use**: No execution errors, production-ready code

## Requirements

### Hardware
- **GPU**: Recommended 16GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: ~20GB for model weights and cache

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU support)

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Dataset Format

The JSON dataset should follow this structure:

```json
[
  {
    "instruction": "Task description or question",
    "input": "Context or specific input",
    "output": "Expected response or answer"
  }
]
```

The example `alzheimers_dataset.json` contains 6 diverse samples covering:
- Risk assessment
- Early warning signs
- Disease progression
- Non-pharmacological interventions
- Cognitive assessment interpretation
- Risk factors

## Configuration

You can modify these parameters in `finetune_mistral_lora.py`:

### Model Configuration
```python
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Base model
MAX_LENGTH = 512  # Maximum sequence length
```

### LoRA Parameters
```python
LORA_R = 16  # Rank (higher = more capacity, more memory)
LORA_ALPHA = 32  # Scaling factor
LORA_DROPOUT = 0.05  # Dropout for regularization
```

### Training Parameters
```python
BATCH_SIZE = 4  # Reduce if out of memory
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
```

## Usage

### Basic Training

```bash
python finetune_mistral_lora.py
```

### Training Output

The script will:
1. Load the dataset (6 examples)
2. Download Mistral-7B model (~14GB)
3. Apply 4-bit quantization
4. Add LoRA adapters
5. Train for 3 epochs
6. Save the fine-tuned model to `./mistral-alzheimers-lora/`

### Expected Output
```
==================================================
Starting Mistral Fine-tuning with LoRA
==================================================

1. Loading dataset...
Loaded 6 training examples

2. Loading model and tokenizer...
Model loaded successfully

3. Preparing model with LoRA...
trainable params: 20,971,520 || all params: 7,261,634,560 || trainable%: 0.2887

4. Setting up training configuration...

5. Creating trainer...

6. Starting training...
==================================================
[Training logs...]

7. Saving model...

==================================================
Training completed successfully!
Model saved to: ./mistral-alzheimers-lora
==================================================
```

## Using the Fine-tuned Model

### Load and Use the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./mistral-alzheimers-lora")
tokenizer = AutoTokenizer.from_pretrained("./mistral-alzheimers-lora")

# Test the model
prompt = """<s>[INST] Analyze the following patient symptoms and provide a risk assessment for Alzheimer's disease.

Patient: 70-year-old with progressive memory loss and confusion. [/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Merge and Save (Optional)

To create a standalone model without needing the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "./mistral-alzheimers-lora")

# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./mistral-alzheimers-merged")
tokenizer.save_pretrained("./mistral-alzheimers-merged")
```

## Expanding the Dataset

To use your own Alzheimer's dataset:

1. Replace `alzheimers_dataset.json` with your data
2. Ensure it follows the same JSON format
3. Adjust `NUM_EPOCHS` and `LEARNING_RATE` based on dataset size:
   - Small dataset (<100): 3-5 epochs
   - Medium dataset (100-1000): 2-3 epochs
   - Large dataset (>1000): 1-2 epochs

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
BATCH_SIZE = 2  # or even 1

# Reduce max length
MAX_LENGTH = 256

# Increase gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 8
```

### Slow Training on CPU
The script will warn you if no GPU is detected. For CPU training:
- Use much smaller model (e.g., Mistral-1B if available)
- Reduce dataset size for testing
- Consider using Google Colab or cloud GPU services

### Model Download Issues
If the model download fails:
```python
# Use mirror or cache
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## Performance Tips

1. **Use fp16/bf16**: Already enabled for faster training
2. **Gradient Checkpointing**: Enabled to save memory
3. **Paged Optimizer**: Using `paged_adamw_32bit` for efficiency
4. **Group by Length**: Enabled for faster training

## Monitoring Training

Enable TensorBoard logging:
```python
# In training_args
report_to="tensorboard"
```

Then run:
```bash
tensorboard --logdir ./mistral-alzheimers-lora/runs
```

## License

This code uses:
- Mistral-7B (Apache 2.0 License)
- Transformers, PEFT, TRL (Apache 2.0)

## Next Steps

1. Expand your dataset with more Alzheimer's-related examples
2. Experiment with different LoRA configurations
3. Try instruction-tuned base models (Mistral-7B-Instruct)
4. Implement evaluation metrics for your specific use case
5. Deploy the model using vLLM or HuggingFace Inference

## Support

For issues or questions:
- Check the error messages carefully
- Verify GPU availability and CUDA installation
- Ensure all dependencies are correctly installed
- Monitor GPU memory usage with `nvidia-smi`
