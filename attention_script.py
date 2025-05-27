import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    # 1. Define Model and Tokenizer names
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    TOKENIZER_NAME = "distilbert-base-uncased" # Compatible tokenizer

    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Loading model: {MODEL_NAME}...")
    # Ensure config explicitly requests attentions if needed, though output_attentions in forward pass is key
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        output_attentions=True # Request attention outputs from the model configuration
    )
    
    # Using CPU
    device = torch.device("cpu")
    model.to(device)
    model.eval() # Set to evaluation mode

    print("Model and tokenizer loaded successfully.")

    # 2. Prepare Single Input
    sentence = "This is a great movie and I really enjoyed it." # A slightly longer sentence
    print(f"\nOriginal sentence: '{sentence}'")

    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device
    
    token_list = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"Tokens: {token_list}")
    print(f"Input tensor shape: {inputs['input_ids'].shape}")

    # 3. Perform Forward Pass
    print("\nPerforming forward pass with output_attentions=True...")
    with torch.no_grad():
        # output_attentions=True in the forward call is often redundant if set in config,
        # but good to be explicit. Some models might only respond to the forward pass argument.
        # However, for DistilBert, config is usually enough. Let's rely on config first.
        outputs = model(**inputs) 

    # 4. Extract and Describe Attention Weights
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        print("\n--- Attention Weights Information ---")
        attentions = outputs.attentions # Tuple of attention tensors, one for each layer
        
        print(f"Type of 'attentions' output: {type(attentions)}")
        print(f"Number of layers (length of attentions tuple): {len(attentions)}")
        
        if len(attentions) > 0:
            first_layer_attentions = attentions[0]
            print(f"Shape of first layer attention tensor: {first_layer_attentions.shape}")
            # Expected shape: (batch_size, num_heads, sequence_length, sequence_length)
            # batch_size = 1
            # num_heads = e.g., 12 for distilbert
            # sequence_length = length of tokenized input sequence
            
            # Optional: Print a small slice to confirm content (e.g., attention of first head for first token)
            # print("\nSlice of first_layer_attentions (first head, first token's attention distribution):")
            # print(first_layer_attentions[0, 0, 0, :]) 
        else:
            print("Attention tuple is empty.")
            
        # The attentions are for DistilBert. It has 6 layers, 12 heads.
        # So, len(attentions) should be 6.
        # Each element attentions[i] should have shape (batch_size, 12, seq_len, seq_len).
        
    else:
        print("\nNo attention weights found in the model output.")
        print(f"Available keys in output: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}")

if __name__ == "__main__":
    main()
