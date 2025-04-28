#!/usr/bin/env python3
"""
Setup script for Mini LLM Project.
Creates the folder structure and initializes empty __init__.py files.
"""
import os
import argparse
from pathlib import Path


def create_directory_structure(base_dir):
    """Create the directory structure for the Mini LLM Project."""
    # Main directories
    directories = [
        "tokenizers",
        "models",
        "training",
        "testing",
        "ui",
        "data",
        "utils",
        "checkpoints",
        "test_results"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
        
        # Create __init__.py in each directory
        init_file = os.path.join(dir_path, "__init__.py")
        with open(init_file, 'w') as f:
            f.write(f'"""\n__init__.py file for {directory} module.\n"""\n')
        print(f"Created file: {init_file}")
    
    # Create empty data directory placeholder
    data_placeholder = os.path.join(base_dir, "data", "README.md")
    with open(data_placeholder, 'w') as f:
        f.write("# Data Directory\n\nPlace your training and testing data files here.\n")
    print(f"Created file: {data_placeholder}")
    
    # Create empty checkpoints directory placeholder
    checkpoints_placeholder = os.path.join(base_dir, "checkpoints", "README.md")
    with open(checkpoints_placeholder, 'w') as f:
        f.write("# Checkpoints Directory\n\nModel checkpoints will be saved here during training.\n")
    print(f"Created file: {checkpoints_placeholder}")
    
    # Create empty test_results directory placeholder
    test_results_placeholder = os.path.join(base_dir, "test_results", "README.md")
    with open(test_results_placeholder, 'w') as f:
        f.write("# Test Results Directory\n\nModel evaluation results will be saved here.\n")
    print(f"Created file: {test_results_placeholder}")
    
    print(f"\nDirectory structure created successfully at: {base_dir}")
    print("You can now copy the Python files into their respective directories.")


def main():
    """Main function to parse arguments and create directory structure."""
    parser = argparse.ArgumentParser(description="Setup script for Mini LLM Project")
    parser.add_argument("--dir", type=str, default="./mini_llm_project",
                        help="Base directory for the project (default: ./mini_llm_project)")
    
    args = parser.parse_args()
    base_dir = args.dir
    
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directory structure
    create_directory_structure(base_dir)


if __name__ == "__main__":
    main()
