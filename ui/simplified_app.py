"""
Simplified Streamlit UI for interacting with the trained language model.
This version avoids device variable scope issues by using consistent device handling.
"""
import streamlit as st
import torch
import os
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path to allow importing from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizers import CharacterTokenizer, BPETokenizer
from models import TransformerModel, DecoderOnlyTransformer


def load_tokenizer(tokenizer_path: str, tokenizer_type: str = "bpe"):
    """
    Load a tokenizer from a saved file.
    
    Args:
        tokenizer_path: Path to the saved tokenizer
        tokenizer_type: Type of tokenizer ('bpe' or 'character')
        
    Returns:
        Loaded tokenizer instance
    """
    if tokenizer_type.lower() == "bpe":
        return BPETokenizer.load(tokenizer_path)
    elif tokenizer_type.lower() == "character":
        return CharacterTokenizer.load(tokenizer_path)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def load_model(model_path: str, model_type: str = "decoder_only"):
    """
    Load a model from a saved file.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ('transformer' or 'decoder_only')
        
    Returns:
        Loaded model instance
    """
    # Always use CPU for simplicity and to avoid device issues
    device = torch.device("cpu")
    
    try:
        # Load model config
        config_path = f"{model_path}_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance
        if model_type.lower() == "transformer":
            model = TransformerModel(config)
        elif model_type.lower() == "decoder_only":
            model = DecoderOnlyTransformer(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model weights
        model_file = f"{model_path}.pt"
        checkpoint = torch.load(model_file, map_location=device)
        
        # Check if the checkpoint contains a state_dict directly or nested under 'model_state_dict'
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # If the checkpoint has a nested structure (saved during training)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # If the checkpoint is a direct state_dict
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Config path: {config_path}, Model path: {model_path}.pt")
        raise e


def main():
    """Main function for the Streamlit UI."""
    st.set_page_config(
        page_title="Mini LLM Project",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Mini LLM Project")
    st.markdown("A small foundational language model built from scratch.")
    
    # Sidebar for model selection and parameters
    st.sidebar.header("Model Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["decoder_only", "transformer"],
        index=0
    )
    
    tokenizer_type = st.sidebar.selectbox(
        "Tokenizer Type",
        ["bpe", "character"],
        index=0
    )
    
    # Generation parameters
    st.sidebar.header("Generation Settings")
    
    max_length = st.sidebar.slider(
        "Maximum Length",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1
    )
    
    top_k = st.sidebar.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=50,
        step=1
    )
    
    # Model loading
    st.sidebar.header("Model Loading")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="./checkpoints/best_model"
    )
    
    tokenizer_path = st.sidebar.text_input(
        "Tokenizer Path",
        value="./data/tokenizer.json"
    )
    
    load_button = st.sidebar.button("Load Model")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Text Generation", "Model Info", "Batch Processing"])
    
    # Initialize session state
    if "model" not in st.session_state:
        st.session_state.model = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "model_info" not in st.session_state:
        st.session_state.model_info = None
    if "device" not in st.session_state:
        st.session_state.device = torch.device("cpu")
    
    # Load model when button is clicked
    if load_button:
        with st.spinner("Loading model and tokenizer..."):
            try:
                # Load tokenizer
                st.session_state.tokenizer = load_tokenizer(tokenizer_path, tokenizer_type)
                
                # Load model and get device
                st.session_state.model, st.session_state.device = load_model(model_path, model_type)
                
                # Get model info
                total_params = sum(p.numel() for p in st.session_state.model.parameters())
                trainable_params = sum(p.numel() for p in st.session_state.model.parameters() if p.requires_grad)
                
                st.session_state.model_info = {
                    "parameter_count": {
                        "total": total_params,
                        "trainable": trainable_params
                    },
                    "config": st.session_state.model.config,
                    "device": st.session_state.device.type
                }
                
                st.session_state.model_loaded = True
                st.success("Model and tokenizer loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.session_state.model_loaded = False
    
    # Text Generation tab
    with tab1:
        st.header("Text Generation")
        
        prompt = st.text_area("Enter a prompt:", height=150)
        generate_button = st.button("Generate Text")
        
        if generate_button:
            if not st.session_state.model_loaded:
                st.warning("Please load a model first.")
            elif not prompt:
                st.warning("Please enter a prompt.")
            else:
                with st.spinner("Generating text..."):
                    try:
                        # Prepare input
                        input_ids = st.session_state.tokenizer.encode(prompt)
                        
                        # Add BOS token if not present
                        if input_ids[0] != st.session_state.tokenizer.token_to_id["[BOS]"]:
                            input_ids = [st.session_state.tokenizer.token_to_id["[BOS]"]] + input_ids
                        
                        # Convert to tensor
                        input_ids = torch.tensor([input_ids], dtype=torch.long).to(st.session_state.device)
                        
                        # Generate text
                        start_time = time.time()
                        with torch.no_grad():
                            output_ids = st.session_state.model.generate(
                                input_ids=input_ids,
                                max_length=max_length,
                                temperature=temperature,
                                top_k=top_k
                            )
                        generation_time = time.time() - start_time
                        
                        # Decode generated text
                        generated_text = st.session_state.tokenizer.decode(output_ids[0].tolist())
                        
                        # Display results
                        st.subheader("Generated Text:")
                        st.write(generated_text)
                        st.info(f"Generation time: {generation_time:.2f} seconds")
                        
                    except Exception as e:
                        st.error(f"Error generating text: {str(e)}")
    
    # Model Info tab
    with tab2:
        st.header("Model Information")
        
        if st.session_state.model_loaded and st.session_state.model_info:
            # Display model architecture
            st.subheader("Model Architecture")
            st.json(st.session_state.model_info["config"])
            
            # Display parameter count
            st.subheader("Parameter Count")
            params = st.session_state.model_info["parameter_count"]
            st.metric("Total Parameters", f"{params['total']:,}")
            st.metric("Trainable Parameters", f"{params['trainable']:,}")
            
            # Display device info
            st.subheader("Runtime Information")
            st.write(f"Device: {st.session_state.model_info['device']}")
            
            # Display tokenizer info
            st.subheader("Tokenizer Information")
            st.write(f"Tokenizer Type: {tokenizer_type}")
            st.write(f"Vocabulary Size: {len(st.session_state.tokenizer.token_to_id)}")
            
            # Display special tokens
            st.write("Special Tokens:")
            special_tokens = {k: v for k, v in st.session_state.tokenizer.token_to_id.items() 
                             if k in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]}
            st.json(special_tokens)
            
        else:
            st.info("Load a model to view its information.")
    
    # Batch Processing tab
    with tab3:
        st.header("Batch Text Processing")
        
        # File upload
        uploaded_file = st.file_uploader("Upload a text file with prompts (one per line)", type=["txt"])
        
        # Process button
        process_button = st.button("Process Batch")
        
        if process_button:
            if not st.session_state.model_loaded:
                st.warning("Please load a model first.")
            elif uploaded_file is None:
                st.warning("Please upload a file.")
            else:
                try:
                    # Read prompts from file
                    prompts = uploaded_file.getvalue().decode("utf-8").splitlines()
                    prompts = [p.strip() for p in prompts if p.strip()]
                    
                    if not prompts:
                        st.warning("No valid prompts found in the file.")
                    else:
                        # Process each prompt
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, prompt in enumerate(prompts):
                            # Update progress
                            progress_bar.progress((i + 1) / len(prompts))
                            
                            # Generate text
                            input_ids = st.session_state.tokenizer.encode(prompt)
                            if input_ids[0] != st.session_state.tokenizer.token_to_id["[BOS]"]:
                                input_ids = [st.session_state.tokenizer.token_to_id["[BOS]"]] + input_ids
                            
                            input_ids = torch.tensor([input_ids], dtype=torch.long).to(st.session_state.device)
                            
                            with torch.no_grad():
                                output_ids = st.session_state.model.generate(
                                    input_ids=input_ids,
                                    max_length=max_length,
                                    temperature=temperature,
                                    top_k=top_k
                                )
                            
                            generated_text = st.session_state.tokenizer.decode(output_ids[0].tolist())
                            
                            results.append({
                                "prompt": prompt,
                                "generated": generated_text
                            })
                        
                        # Display results
                        st.subheader("Batch Results:")
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1}: {result['prompt'][:50]}..."):
                                st.write("**Prompt:**")
                                st.write(result["prompt"])
                                st.write("**Generated:**")
                                st.write(result["generated"])
                        
                        # Download results
                        result_json = json.dumps(results, indent=2)
                        st.download_button(
                            label="Download Results",
                            data=result_json,
                            file_name="batch_results.json",
                            mime="application/json"
                        )
                        
                except Exception as e:
                    st.error(f"Error processing batch: {str(e)}")


if __name__ == "__main__":
    main()
