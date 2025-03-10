#!/usr/bin/env python3
# Email Responder LLM Test Script
# Selects a random .eml file from the correspondence folder and generates a response

import os
import sys
import random
import subprocess
from pathlib import Path

def find_eml_files(directory):
    """Find all .eml files in the specified directory"""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    eml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.eml'):
                eml_files.append(os.path.join(root, file))
    
    if not eml_files:
        print(f"Error: No .eml files found in '{directory}'")
        sys.exit(1)
    
    return eml_files

def main():
    # Define the correspondence directory
    correspondence_dir = Path("training_emails/correspondence")
    
    # Find all .eml files
    eml_files = find_eml_files(correspondence_dir)
    
    # Select a random .eml file
    random_file = random.choice(eml_files)
    
    print(f"\n{'='*80}")
    print(f"Selected file: {random_file}")
    print(f"{'='*80}\n")
    
    try:
        # Call the email_responder_llm.py script
        result = subprocess.run(
            ["python", "email_responder_llm.py", "--email", random_file, "--provider", "anthropic"], 
            capture_output=True, 
            text=True,
            check=True
        )
        
        # Print the response
        print(f"{'='*80}")
        print("Generated Response:")
        print(f"{'='*80}\n")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running email_responder_llm.py: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()

