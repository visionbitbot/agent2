import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import time

# 1. Specify model and dataset names
PRE_FINETUNED_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
TOKENIZER_MODEL_NAME = "distilbert-base-uncased" # Tokenizer is compatible

DATASET_NAME = "glue"
DATASET_SUBSET = "sst2"
BATCH_SIZE = 32 
VALIDATION_SUBSET_SIZE = 32 # Using a very small subset (1 batch)

# 2. Load tokenizer
print(f"Loading tokenizer: '{TOKENIZER_MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

# 3. Load dataset
print(f"Loading dataset: '{DATASET_NAME}/{DATASET_SUBSET}'...")
raw_datasets = load_dataset(DATASET_NAME, DATASET_SUBSET)

# 4. Implement preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

# 5. Apply preprocessing to a VERY SMALL SUBSET of the validation set
print(f"Preprocessing a subset of {VALIDATION_SUBSET_SIZE} samples from the validation set...")
if 'validation' not in raw_datasets:
    print("Error: 'validation' split not found in the dataset.")
    exit()

validation_data_full = raw_datasets['validation']
actual_validation_subset_size = min(VALIDATION_SUBSET_SIZE, len(validation_data_full))
if actual_validation_subset_size == 0:
    print("Error: Validation subset size is 0. Cannot proceed.")
    exit()
validation_subset = validation_data_full.select(range(actual_validation_subset_size))

tokenized_validation_subset = validation_subset.map(preprocess_function, batched=True)
print(f"Number of samples in tokenized validation subset: {len(tokenized_validation_subset)}")

print("\nRaw dataset information:")
for split_name, data in raw_datasets.items():
    print(f"Number of samples in raw {split_name}: {len(data)}")


# 6. Load the Pre-Fine-tuned Model
print(f"\nLoading pre-fine-tuned model: '{PRE_FINETUNED_MODEL_NAME}'...")
model = AutoModelForSequenceClassification.from_pretrained(PRE_FINETUNED_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model '{PRE_FINETUNED_MODEL_NAME}' loaded successfully.")
print(f"Using device: {device}")


# --- Evaluation of the Pre-Fine-tuned Model on a Subset ---

# 7. Define Evaluation Function
def evaluate_model(model, dataloader, device, eval_type="Pre-fine-tuned Model on Subset"):
    model.eval()
    all_preds = []
    all_labels = []
    start_time = time.time()
    
    dataset_size = len(dataloader.dataset)
    print(f"\nStarting evaluation for: {eval_type} ({dataset_size} samples)")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"  Evaluating batch {i+1}/{len(dataloader)}...") # Print for every batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) 
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    end_time = time.time()
    total_time = end_time - start_time
    accuracy = accuracy_score(all_labels, all_preds)
    
    avg_inference_time_per_sample = total_time / dataset_size if dataset_size > 0 else 0
    
    print(f"\n--- {eval_type} Evaluation Results ---")
    print(f"Validation Subset Accuracy: {accuracy:.4f}")
    print(f"Total Inference Time: {total_time:.2f} seconds for {dataset_size} samples")
    print(f"Average Inference Time per Sample: {avg_inference_time_per_sample:.6f} seconds")
    return accuracy, total_time

# 8. Prepare Validation DataLoader for the SUBSET
tokenized_validation_subset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Adjust batch size if subset size is smaller than BATCH_SIZE
current_batch_size = BATCH_SIZE
if actual_validation_subset_size < BATCH_SIZE:
    current_batch_size = actual_validation_subset_size

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
validation_subset_dataloader = DataLoader(
    tokenized_validation_subset, 
    batch_size=current_batch_size, # Use potentially adjusted batch size
    collate_fn=data_collator
)
    
# 9. Perform Evaluation of the Pre-Fine-tuned Model on the SUBSET
eval_label = f"Pre-fine-tuned DistilBERT (sst-2-english) on {actual_validation_subset_size}-sample Validation Subset"
evaluate_model(model, validation_subset_dataloader, device, eval_type=eval_label)

print("\nScript execution complete.")
