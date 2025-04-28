# Mini LLM Django UI Documentation

This document provides instructions for setting up and using the Django UI for the Mini LLM project.

## Overview

The Mini LLM Django UI is a web-based interface that replaces the command-line functionality of the original `main.py` script. It provides a user-friendly way to:

- Upload and manage training data
- Create and configure tokenizers
- Create and configure models
- Train new models from scratch
- Upload and retrain existing models
- Track training progress in real-time
- Generate text with trained models

## Installation

### Prerequisites

- Python 3.8 or higher
- Django 3.2 or higher
- Channels (for WebSocket support)

### Optional Dependencies

The following dependencies are only required when actually running model training or text generation:

- PyTorch
- NumPy
- The original Mini LLM project code

### Setup Instructions

1. Clone the repository or extract the provided zip file:

```bash
cd /path/to/your/projects
unzip mini_llm_django.zip
cd mini_llm_django
```

2. Install the required dependencies:

```bash
pip install django channels
```

3. Create the database:

```bash
python manage.py makemigrations llm_app
python manage.py migrate
```

4. Start the development server:

```bash
python manage.py runserver
```

5. Access the UI in your web browser at: http://127.0.0.1:8000/

## Using the Django UI

The Mini LLM Django UI is a single-page application with tabs for different functionality.

### Dashboard

The dashboard provides an overview of your Mini LLM project, including:

- Number of available datasets
- Number of configured models
- Number of training sessions
- Recent training sessions
- Recent text generations

### Training Data

In this tab, you can:

1. **Upload Training Data**: Use the form to upload text files containing your training data.
2. **View Available Data**: See a list of all uploaded training data files with their size and upload date.
3. **Delete Data**: Remove training data files that are no longer needed.

### Tokenizers

In this tab, you can:

1. **Create/Upload Tokenizers**: Create new tokenizers or upload existing ones.
   - For new tokenizers, select the type (BPE or Character) and set the vocabulary size.
   - For existing tokenizers, upload the tokenizer file.
2. **View Available Tokenizers**: See a list of all tokenizers with their type and vocabulary size.
3. **Delete Tokenizers**: Remove tokenizers that are no longer needed.

### Models

In this tab, you can:

1. **Create New Models**: Configure a new model with the following parameters:
   - Name
   - Model Type (Transformer, Decoder-Only, or Encoder-Only)
   - Tokenizer
   - Model dimensions (d_model, num_heads, num_layers, d_ff)
   - Maximum sequence length
   - Dropout rate
2. **Upload Existing Models**: Upload a pre-trained model with its checkpoint and configuration files.
3. **View Available Models**: See a list of all models with their type, tokenizer, and training status.
4. **Train Models**: Start training a model by clicking the "Train" button.
5. **Delete Models**: Remove models that are no longer needed.

### Training

In this tab, you can:

1. **View Training Sessions**: See a list of all training sessions with their status, start/end times, and metrics.
2. **View Training Progress**: Click "View Progress" to see real-time training progress with charts for loss and perplexity.
3. **Configure Training**: When starting a new training session, configure:
   - Training data
   - Batch size
   - Learning rate
   - Number of epochs
   - Early stopping patience
   - Validation split

### Text Generation

In this tab, you can:

1. **Generate Text**: Select a trained model and enter a prompt to generate text.
2. **Configure Generation**: Set parameters like maximum length, temperature, and top-k.
3. **View Generated Text**: See the generated text in the output area.

## Integration with Existing Mini LLM Project

The Django UI is designed to work with the existing Mini LLM project code. It expects the Mini LLM project directory to be located in the parent directory of the Django project.

For example:
```
/home/user/projects/
├── mini_llm_project/
│   ├── tokenizers/
│   ├── models/
│   ├── training/
│   └── ...
└── mini_llm_django/
    ├── llm_app/
    ├── mini_llm_ui/
    └── ...
```

If your Mini LLM project is located elsewhere, you'll need to modify the `sys.path.append()` line in `training_manager.py` to point to the correct location.

## Dependency Management

The Django UI is designed to work even without PyTorch and other heavy ML dependencies installed. This allows you to set up and configure the UI on systems with limited resources.

When you attempt to run training or text generation without the required dependencies, the UI will display an error message indicating which dependencies are missing.

To enable full functionality, install the required ML dependencies:

```bash
pip install torch numpy
```

## Troubleshooting

### Database Issues

If you encounter database errors, try resetting the database:

```bash
rm db.sqlite3
python manage.py makemigrations llm_app
python manage.py migrate
```

### Missing Dependencies

If you see errors about missing dependencies during training or text generation, install the required packages:

```bash
pip install torch numpy
```

### WebSocket Connection Issues

If real-time training updates aren't working:

1. Make sure you're using a modern browser that supports WebSockets
2. Check that the Channels package is installed correctly
3. Restart the Django development server

## Extending the UI

The Mini LLM Django UI is designed to be modular and extensible. To add new features:

1. Add new models in `models.py`
2. Create forms in `forms.py`
3. Add views in `views.py` or `training_views.py`
4. Update the templates in `templates/llm_app/`
5. Add JavaScript functionality in `static/llm_app/js/main.js`

## License

This project is licensed under the same license as the original Mini LLM project.
