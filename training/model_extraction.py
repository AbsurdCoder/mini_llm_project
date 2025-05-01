"""
Utilities for loading, saving, and extracting model components.
Handles both custom models and Hugging Face models/tokenizers.
"""
import os
import json
import logging
import torch

# Try importing Hugging Face transformers
try:
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, AutoConfig = None, None, None, None

# Assuming your custom models are defined elsewhere and can be imported
# from ..models.transformer_model import TransformerModel # Example
# from ..models.decoder_model import DecoderModel # Example
# from ..models.encoder_model import EncoderModel # Example
# from ..tokenizers.bpe_tokenizer import BPETokenizer # Example
# from ..tokenizers.char_tokenizer import CharTokenizer # Example

# Placeholder for custom model/tokenizer loading logic
# In a real scenario, these would import and instantiate your custom classes
def load_custom_model_from_config(config, model_type):
    # This function needs to be implemented based on your custom model structure
    # It should take the config dict and model_type, and return an instantiated model
    logger = logging.getLogger(__name__)
    logger.warning(f"Placeholder: load_custom_model_from_config called for type {model_type}. Needs implementation.")
    # Example (replace with your actual model loading):
    # if model_type == 'transformer':
    #     return TransformerModel(**config)
    # elif model_type == 'decoder_only':
    #     return DecoderModel(**config)
    # elif model_type == 'encoder_only':
    #     return EncoderModel(**config)
    # else:
    #     raise ValueError(f"Unknown custom model type: {model_type}")
    # Return a dummy nn.Module for now if not implemented
    return torch.nn.Module()

def load_custom_tokenizer_from_path(path):
    # This function needs to be implemented based on your custom tokenizer structure
    # It should load the tokenizer state (e.g., vocab) from the given path
    logger = logging.getLogger(__name__)
    logger.warning(f"Placeholder: load_custom_tokenizer_from_path called for path {path}. Needs implementation.")
    # Example (replace with your actual tokenizer loading):
    # if path.endswith('.json'): # Assuming BPETokenizer saves/loads from json
    #     try:
    #         tokenizer = BPETokenizer(vocab_path=None) # Initialize empty
    #         tokenizer.load(path) # Load state
    #         return tokenizer
    #     except Exception as e:
    #         logger.error(f"Failed to load custom BPE tokenizer from {path}: {e}")
    #         raise
    # else: # Add logic for other custom tokenizers like CharTokenizer
    #     raise ValueError(f"Unknown custom tokenizer format/type at path: {path}")
    # Return a dummy object for now
    class DummyTokenizer:
        def encode(self, text): return [0]
        def decode(self, ids): return ""
        vocab_size = 1
        pad_token_id = 0
    return DummyTokenizer()

def load_model_and_tokenizer(
    model_path: str = None,
    tokenizer_path: str = None,
    hf_model_name: str = None,
    hf_tokenizer_name: str = None,
    offline_hf_dir: str = None,
    device: str = 'cpu',
    logger: logging.Logger = None,
    model_type: str = None):
    """
    Loads a model and tokenizer, supporting both custom checkpoints and Hugging Face models.

    Args:
        model_path (str, optional): Path to the custom model checkpoint (.pt) or HF model directory.
        tokenizer_path (str, optional): Path to the custom tokenizer file or HF tokenizer directory.
        hf_model_name (str, optional): Name of the Hugging Face model (e.g., 'gpt2', 'distilbert-base-uncased').
        hf_tokenizer_name (str, optional): Name of the Hugging Face tokenizer. Defaults to hf_model_name if None.
        offline_hf_dir (str, optional): Local directory containing pre-downloaded Hugging Face model/tokenizer files.
        device (str): Device to load the model onto ('cpu', 'cuda', 'mps').
        logger (logging.Logger, optional): Logger instance.
        model_type (str, optional): Type of the custom model ('transformer', 'decoder_only', 'encoder_only'), 
                                    required if loading a custom model from a checkpoint without config.

    Returns:
        tuple: (model, tokenizer, config, is_hf_model)
               Returns None for model/tokenizer if loading fails.
               Config is the model configuration object/dict.
               is_hf_model is a boolean flag.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    model = None
    tokenizer = None
    config = None
    is_hf_model = False

    # --- Hugging Face Model/Tokenizer Loading ---


    if HUGGINGFACE_AVAILABLE and (hf_model_name or (offline_hf_dir and os.path.exists(offline_hf_dir))):
        is_hf_model = True
        load_path = offline_hf_dir if offline_hf_dir and os.path.exists(offline_hf_dir) else hf_model_name
        tokenizer_load_path = offline_hf_dir if offline_hf_dir and os.path.exists(offline_hf_dir) else (hf_tokenizer_name or hf_model_name)
        
        logger.info(f"Loading Hugging Face model from: {load_path}")
        logger.info(f"Loading Hugging Face tokenizer from: {tokenizer_load_path}")
        
        try:
            # Load Config
            config = AutoConfig.from_pretrained(load_path)
            # Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
            # Ensure pad token is set if missing (common issue with GPT-2)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    logger.warning("Tokenizer missing pad token, setting it to eos_token")
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    logger.warning("Tokenizer missing pad_token and eos_token. Adding a default pad token '[PAD]'")
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load Model (Try CausalLM first, fallback to MaskedLM if needed, or use AutoModel)
            # This might need refinement based on the exact model types you expect
            try:
                model = AutoModelForCausalLM.from_pretrained(load_path, config=config)
                logger.info(f"Loaded as Causal LM: {model.__class__.__name__}")
            except Exception as e_causal:
                logger.warning(f"Could not load as Causal LM ({e_causal}), trying Masked LM...")
                try:
                    model = AutoModelForMaskedLM.from_pretrained(load_path, config=config)
                    logger.info(f"Loaded as Masked LM: {model.__class__.__name__}")
                except Exception as e_masked:
                    logger.error(f"Failed to load HF model as Causal or Masked LM: {e_masked}")
                    return None, None, None, False
            
            model.to(device)
            # Attach config to model if not already there (HF usually does this)
            if not hasattr(model, 'config'):
                model.config = config
                
        except Exception as e:
            logger.error(f"Error loading Hugging Face model/tokenizer from {load_path}: {e}")
            return None, None, None, False

    # --- Custom Model/Tokenizer Loading --- 
    elif model_path:
        logger.info(f"Loading custom model from checkpoint: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Custom model checkpoint file not found: {model_path}")
            return None, None, {}, False # Return empty config dict
            
        config = {} # Default config to empty dict
        model = None
        tokenizer = None
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # --- Config Loading ---
            if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                config = checkpoint['config']
                logger.info("Loaded model config from checkpoint key 'config'.")
            elif 'model_hyperparams' in checkpoint and isinstance(checkpoint['model_hyperparams'], dict): # Legacy check?
                config = checkpoint['model_hyperparams']
                logger.info("Loaded model config from checkpoint key 'model_hyperparams'.")
            else:
                logger.warning("Model config dictionary not found in checkpoint. Using empty config dict.")
            
            # --- Model Type Determination ---
            if not model_type:
                if isinstance(config, dict) and 'model_type' in config:
                    model_type = config['model_type']
                elif 'model_type' in checkpoint and isinstance(checkpoint['model_type'], str):
                     model_type = checkpoint['model_type']
                else:
                    logger.error("Cannot determine custom model type from checkpoint or args. Please provide --model_type.")
                    return None, None, config, False # Return config dict found so far
            logger.info(f"Inferred/Using custom model type: {model_type}")

            # --- Model Instantiation ---
            try:
                # !!! IMPORTANT: Replace load_custom_model_from_config with actual model imports and instantiation !!!
                # This placeholder will likely cause issues later if not replaced.
                from models.transformer_model import TransformerModel, DecoderOnlyTransformer # Example - Adjust imports!
                from models.encoder_model import EncoderOnlyModel # Example - Adjust imports!
                if model_type == "transformer":
                    model = TransformerModel(config=config)
                elif model_type == "decoder_only":
                    model = DecoderOnlyTransformer(config=config)
                elif model_type == "encoder_only":
                    model = EncoderOnlyModel(config=config)
                else:
                     raise ValueError(f"Unsupported custom model type: {model_type}")
                # model = load_custom_model_from_config(config, model_type) # Original placeholder call
                model.to(device)
                logger.info(f"Instantiated custom model: {model.__class__.__name__}")
            except Exception as model_inst_e:
                 logger.error(f"Failed to instantiate custom model type '{model_type}': {model_inst_e}")
                 logger.error("Ensure the correct model classes are imported in model_extraction.py and config is valid.")
                 return None, None, config, False # Return config dict

            # --- State Dict Loading ---
            state_dict_key = None
            loaded_state_dict = None
            if 'model_state_dict' in checkpoint:
                 state_dict_key = 'model_state_dict'
                 loaded_state_dict = checkpoint[state_dict_key]
            elif 'state_dict' in checkpoint:
                 state_dict_key = 'state_dict'
                 loaded_state_dict = checkpoint[state_dict_key]
            # Check if checkpoint itself is the state_dict (and not our structured dict)
            elif all(isinstance(k, str) for k in checkpoint.keys()) and 'epoch' not in checkpoint:
                 logger.info("Checkpoint appears to be a raw state_dict.")
                 loaded_state_dict = checkpoint
                 
            if loaded_state_dict:
                logger.info(f"Loading model state_dict... (Source key: '{state_dict_key if state_dict_key else 'root'}')")
                try:
                    model.load_state_dict(loaded_state_dict)
                except RuntimeError as e:
                     logger.error(f"Failed to load state_dict: {e}.\nCheckpoint keys: {list(loaded_state_dict.keys())[:10]}...\nModel expects keys like: {list(model.state_dict().keys())[:40]}...")
                     return model, None, config, False # Return model instance but indicate failure
            else:
                 logger.error("Could not find a valid model state_dict in the checkpoint.")
                 return model, None, config, False # Return model instance but indicate failure

            # Attach config to model if not done during instantiation (should be redundant now)
            if not hasattr(model, 'config'):
                model.config = config
                 
            # --- Tokenizer Loading ---
            # !!! IMPORTANT: Replace load_custom_tokenizer_from_path with actual tokenizer imports and loading !!!
            if tokenizer_path:
                logger.info(f"Loading custom tokenizer from: {tokenizer_path}")
                if not os.path.exists(tokenizer_path):
                    logger.error(f"Custom tokenizer file not found: {tokenizer_path}. Proceeding without tokenizer.")
                    tokenizer = None
                else:
                    try:
                        # tokenizer = load_custom_tokenizer_from_path(tokenizer_path) # Original placeholder call
                        from ctokenizers.bpe_tokenizer import BPETokenizer # Example
                        from ctokenizers.character_tokenizer import CharacterTokenizer # Example
                        # Infer tokenizer type from path or checkpoint info if possible
                        # For now, assume json -> BPE
                        if tokenizer_path.endswith('.json'):
                             tokenizer = BPETokenizer.load(tokenizer_path)
                             logger.info(f"Loaded custom tokenizer: {tokenizer.__class__.__name__}")
                        else:
                             logger.warning(f"Cannot determine tokenizer type from path: {tokenizer_path}. Attempting BPE load.")
                             try:
                                 tokenizer = BPETokenizer.load(tokenizer_path)
                                 logger.info(f"Loaded custom tokenizer: {tokenizer.__class__.__name__}")
                             except Exception as bpe_e:
                                 logger.error(f"Failed to load custom tokenizer as BPE: {bpe_e}")
                                 tokenizer = None
                    except Exception as e:
                        logger.error(f"Failed to load custom tokenizer from {tokenizer_path}: {e}")
                        tokenizer = None # Proceed without tokenizer
            else:
                logger.warning("No custom tokenizer path provided. Proceeding without tokenizer.")
                tokenizer = None
                
        except Exception as e:
            logger.exception(f"Critical error loading custom model/tokenizer from {model_path}: {e}")
            # Ensure config is at least an empty dict before returning
            if config is None: config = {}
            return None, None, config, False # Return config dict
            
    else:
        logger.error("Must provide either Hugging Face info (name or offline dir) or a custom model path.")
        return None, None, None, False

    logger.info(f"Model and Tokenizer loading complete. Model: {model.__class__.__name__ if model else 'None'}, Tokenizer: {tokenizer.__class__.__name__ if tokenizer else 'None'}")
    return model, tokenizer, config, is_hf_model

def save_model_and_tokenizer(
    model,
    tokenizer,
    save_directory: str,
    is_hf_model: bool,
    logger: logging.Logger = None
):
    """
    Saves the model and tokenizer.

    Args:
        model: The model instance.
        tokenizer: The tokenizer instance.
        save_directory (str): The directory to save the model and tokenizer.
        is_hf_model (bool): Flag indicating if it's a Hugging Face model/tokenizer.
        logger (logging.Logger, optional): Logger instance.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    os.makedirs(save_directory, exist_ok=True)

    try:
        if is_hf_model and HUGGINGFACE_AVAILABLE:
            logger.info(f"Saving Hugging Face model and tokenizer to: {save_directory}")
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)
        else:
            # Custom saving logic
            logger.info(f"Saving custom model and tokenizer to: {save_directory}")
            
            # Save model state dictionary
            model_save_path = os.path.join(save_directory, "custom_model.pt")
            save_obj = {
                'model_state_dict': model.state_dict(),
                'config': getattr(model, 'config', {}), # Save config if attached
                'model_type': getattr(model, 'model_type', None) # Save type if attached
            }
            torch.save(save_obj, model_save_path)
            logger.info(f"Custom model state saved to {model_save_path}")
            print(save_obj)
            # Save tokenizer (assuming a .save() method)
            if hasattr(tokenizer, 'save'):
                tokenizer_save_path = os.path.join(save_directory, "custom_tokenizer.json") # Or other format
                try:
                    tokenizer.save(tokenizer_save_path)
                    logger.info(f"Custom tokenizer saved to {tokenizer_save_path}")
                except Exception as e:
                    logger.error(f"Failed to save custom tokenizer: {e}")
            else:
                logger.warning("Custom tokenizer does not have a .save() method. Cannot save tokenizer.")
                
    except Exception as e:
        logger.exception(f"Error saving model/tokenizer to {save_directory}: {e}")
        raise # Re-raise the exception after logging

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # --- Test HF Loading (Online) ---
    logger.info("\n--- Testing HF Online Loading ---")
    # model_hf, tokenizer_hf, config_hf, is_hf = load_model_and_tokenizer(hf_model_name='gpt2', device='cpu', logger=logger)
    # if model_hf:
    #     logger.info("HF Online Load Successful.")
    #     # Test saving
    #     # save_model_and_tokenizer(model_hf, tokenizer_hf, "./test_hf_save", is_hf, logger)
    #     # logger.info("HF Save Successful.")
    # else:
    #     logger.error("HF Online Load Failed.")

    # --- Test HF Loading (Offline) --- 
    # Requires downloading 'gpt2' to './hf_offline_cache/gpt2' first using hf_offline_downloader.py
    logger.info("\n--- Testing HF Offline Loading ---")
    # model_hf_off, tok_hf_off, cfg_hf_off, is_hf_off = load_model_and_tokenizer(
    #     offline_hf_dir='./hf_offline_cache/gpt2', # Assumes model is here
    #     device='cpu', 
    #     logger=logger
    # )
    # if model_hf_off:
    #     logger.info("HF Offline Load Successful.")
    # else:
    #     logger.error("HF Offline Load Failed. (Did you download the model first?)")

    # --- Test Custom Loading (Placeholder) ---
    logger.info("\n--- Testing Custom Loading (Placeholder) ---")
    # Need a dummy checkpoint and tokenizer file for this to run
    # os.makedirs("./test_custom_save", exist_ok=True)
    # dummy_checkpoint = {'model_state_dict': {}, 'config': {'model_type': 'decoder_only', 'vocab_size': 10}}
    # torch.save(dummy_checkpoint, "./test_custom_save/custom_model.pt")
    # with open("./test_custom_save/custom_tokenizer.json", 'w') as f: json.dump({'vocab': {'<pad>': 0}}, f)
    
    # model_c, tok_c, cfg_c, is_c = load_model_and_tokenizer(
    #     model_path="./test_custom_save/custom_model.pt", 
    #     tokenizer_path="./test_custom_save/custom_tokenizer.json", 
    #     device='cpu', 
    #     logger=logger,
    #     model_type='decoder_only' # Specify type if not in config/checkpoint
    # )
    # if model_c:
    #     logger.info("Custom Load Successful (Placeholder). Note: Requires implementing load_custom_model_from_config and load_custom_tokenizer_from_path.")
    #     # Test saving
    #     # save_model_and_tokenizer(model_c, tok_c, "./test_custom_save_again", is_c, logger)
    #     # logger.info("Custom Save Successful (Placeholder).")
    # else:
    #     logger.error("Custom Load Failed (Placeholder).")





from .enhanced_detailed_trainer import EnhancedTrainer # Add this import

def continue_training(
    model,
    tokenizer,
    config,
    train_texts,
    val_texts,
    args, # Contains all command-line args like lr, epochs, paths etc.
    device,
    logger
):
    """
    Sets up and runs the EnhancedTrainer to continue training an existing model.

    Args:
        model: Loaded model instance.
        tokenizer: Loaded tokenizer instance.
        config: Loaded model configuration.
        train_texts: List of new training texts.
        val_texts: List of new validation texts.
        args: Namespace object containing all command-line arguments.
        device: Device string ("cpu", "cuda", "mps").
        logger: Logger instance.
    """
    logger.info("Setting up trainer for continued training...")
    
    # Determine if it's an HF model from the config
    is_hf_model = config.get("is_hf_model", False)
    
    # Extract relevant args for the trainer
    # Note: We might need to load previous optimizer/scheduler states if the checkpoint contains them
    # For simplicity now, we re-initialize them based on args.
    start_epoch = 0 # Assuming we restart epoch count, or load from checkpoint if available
    # if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
    
    
    trainer = EnhancedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts, 
        val_texts=val_texts,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs, # Use total epochs from args for continuation
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        model_save_path=args.model_path, # Use the same base path for saving best model
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        max_seq_len=args.max_seq_len, 
        verbose=args.verbose,
        is_hf_model=is_hf_model,
        start_epoch=start_epoch # Pass start_epoch if resuming state
    )
    
    # Start the training process
    logger.info("Starting continued training process...")
    trainer.train()
    logger.info("Continued training finished.")


