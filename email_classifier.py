#!/usr/bin/env python3
"""
Email Classifier

This script classifies emails into two categories:
- 'correspondence': Emails where a response is expected/required
- 'notifications': Emails where no response is expected (advertisements, statements, etc.)

It uses TF-IDF vectorization with a RandomForest classifier from scikit-learn.

Usage:
    # Train from directory structure
    python email_classifier.py train --dir path/to/emails --output model.pkl

    # Train from CSV file
    python email_classifier.py train --csv labels.csv --emails path/to/emails --output model.pkl

    # Evaluate model
    python email_classifier.py evaluate --model model.pkl --test path/to/test_emails

    # Classify multiple emails
    python email_classifier.py classify --model model.pkl --emails path/to/emails_to_classify

    # Classify a single email (outputs only the classification)
    python email_classifier.py classify --model model.pkl --email path/to/single/email.eml
"""

import os
import re
import csv
import argparse
import pickle
import datetime
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class EmailClassifier:
    """
    A class to classify emails as either 'correspondence' or 'notifications'.
    """
    
    def __init__(self):
        """Initialize the classifier pipeline with TF-IDF and RandomForest."""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.85,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
        self.categories = ['correspondence', 'notifications']
    
    def extract_email_features(self, email_path):
        """
        Extract relevant features from an email file.
        
        Args:
            email_path: Path to the email file
            
        Returns:
            str: Processed email text combining subject and body
        """
        try:
            with open(email_path, 'rb') as fp:
                msg = BytesParser(policy=default).parse(fp)
            
            # Extract the subject
            subject = msg.get('subject', '')
            
            # Extract the body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode('utf-8', errors='ignore')
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
            
            # Additional metadata that might be helpful
            from_field = msg.get('from', '')
            
            # Combine features into a single text
            combined_text = f"Subject: {subject}\nFrom: {from_field}\n\n{body}"
            
            # Clean text
            cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
            
            return cleaned_text
        except Exception as e:
            print(f"Error processing email {email_path}: {e}")
            return ""
    
    def load_from_directory(self, dir_path):
        """
        Load emails from directories named by their categories.
        
        Args:
            dir_path: Path to the directory containing category subdirectories
            
        Returns:
            tuple: (X, y) where X is a list of email texts and y is a list of labels
        """
        X, y = [], []
        
        for category in self.categories:
            category_dir = os.path.join(dir_path, category)
            if not os.path.isdir(category_dir):
                continue
                
            for filename in os.listdir(category_dir):
                file_path = os.path.join(category_dir, filename)
                if os.path.isfile(file_path):
                    email_text = self.extract_email_features(file_path)
                    if email_text:
                        X.append(email_text)
                        y.append(category)
        
        return X, y
    
    def load_from_csv(self, csv_path, emails_dir):
        """
        Load emails and labels from a CSV file.
        
        Args:
            csv_path: Path to the CSV file with labels
            emails_dir: Path to the directory containing the email files
            
        Returns:
            tuple: (X, y) where X is a list of email texts and y is a list of labels
        """
        X, y = [], []
        
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 2:
                    email_file, label = row[0], row[1]
                    if label in self.categories:
                        file_path = os.path.join(emails_dir, email_file)
                        if os.path.isfile(file_path):
                            email_text = self.extract_email_features(file_path)
                            if email_text:
                                X.append(email_text)
                                y.append(label)
        
        return X, y
    
    def train(self, X, y):
        """
        Train the classifier.
        
        Args:
            X: List of email texts
            y: List of corresponding labels
            
        Returns:
            self: The trained classifier instance
        """
        self.pipeline.fit(X, y)
        return self
    
    def evaluate(self, X, y):
        """
        Evaluate the classifier performance.
        
        Args:
            X: List of email texts
            y: List of corresponding labels
            
        Returns:
            dict: Dictionary with accuracy and classification report
        """
        y_pred = self.pipeline.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=self.categories)
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    def classify(self, X):
        """
        Classify email texts.
        
        Args:
            X: List of email texts
            
        Returns:
            list: Predicted labels for each email
        """
        return self.pipeline.predict(X)
    
    def save(self, model_path):
        """
        Save the model to a file.
        
        Args:
            model_path: Path where to save the model
        """
        # Check if the model file already exists
        if os.path.exists(model_path):
            # Get creation time and format it as a string
            ctime = os.path.getctime(model_path)
            timestamp = datetime.datetime.fromtimestamp(ctime).strftime('%Y%m%d_%H%M%S')
            
            # Create backup filename with timestamp
            backup_path = f"{model_path}.{timestamp}.bak"
            
            # Rename the existing file to the backup name
            os.rename(model_path, backup_path)
            print(f"Created backup of existing model at: {backup_path}")
        
        # Save the new model
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, model_path):
        """
        Load a model from a file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            EmailClassifier: Loaded classifier instance
        """
        with open(model_path, 'rb') as f:
            return pickle.load(f)


def get_emails_from_dir(directory):
    """
    Get all email files from a directory.
    
    Args:
        directory: Path to the directory
        
    Returns:
        list: List of email file paths
    """
    email_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            email_paths.append(file_path)
    
    return email_paths


def main():
    """Parse command-line arguments and execute the requested command."""
    parser = argparse.ArgumentParser(description='Email Classifier')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train the classifier')
    train_parser.add_argument('--dir', help='Directory with category subdirectories')
    train_parser.add_argument('--csv', help='CSV file with email labels')
    train_parser.add_argument('--emails', help='Directory with email files (used with --csv)')
    train_parser.add_argument('--output', required=True, help='Output model file')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    
    # Evaluate parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the classifier')
    eval_parser.add_argument('--model', required=True, help='Path to the trained model')
    eval_parser.add_argument('--test', required=True, help='Directory with test emails')
    
    # Classify parser
    classify_parser = subparsers.add_parser('classify', help='Classify emails')
    classify_parser.add_argument('--model', required=True, help='Path to the trained model')
    classify_parser.add_argument('--emails', help='Directory with emails to classify')
    classify_parser.add_argument('--email', help='Path to a single email file to classify')
    classify_parser.add_argument('--output', help='Output CSV file for classification results')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        classifier = EmailClassifier()
        
        # Load data from directory or CSV
        if args.dir:
            print(f"Loading emails from directory: {args.dir}")
            X, y = classifier.load_from_directory(args.dir)
        elif args.csv and args.emails:
            print(f"Loading emails from CSV: {args.csv} and directory: {args.emails}")
            X, y = classifier.load_from_csv(args.csv, args.emails)
        else:
            parser.error("Either --dir or --csv and --emails must be provided")
        
        if not X or not y:
            print("No emails loaded. Check your input paths.")
            return
        
        print(f"Loaded {len(X)} emails for training")
        
        # Split data for training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        # Train the classifier
        print("Training classifier...")
        classifier.train(X_train, y_train)
        
        # Evaluate on the test set
        print("Evaluating classifier...")
        results = classifier.evaluate(X_test, y_test)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(results['report'])
        
        # Save the model (with backup if needed)
        classifier.save(args.output)
        print(f"Model saved to: {args.output}")
        
    elif args.command == 'evaluate':
        # Load the model
        classifier = EmailClassifier.load(args.model)
        
        # Load test data
        X_test, y_test = [], []
        for category in classifier.categories:
            category_dir = os.path.join(args.test, category)
            if os.path.isdir(category_dir):
                for filename in os.listdir(category_dir):
                    file_path = os.path.join(category_dir, filename)
                    if os.path.isfile(file_path):
                        email_text = classifier.extract_email_features(file_path)
                        if email_text:
                            X_test.append(email_text)
                            y_test.append(category)
        
        if not X_test or not y_test:
            print("No test emails found. Check your test directory.")
            return
        
        print(f"Loaded {len(X_test)} emails for evaluation")
        
        # Evaluate the classifier
        results = classifier.evaluate(X_test, y_test)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(results['report'])
        
    elif args.command == 'classify':
        # Load the model
        classifier = EmailClassifier.load(args.model)
        
        # Check if we're classifying a single email or multiple emails
        if args.email:
            # Single email classification
            email_path = args.email
            if not os.path.isfile(email_path):
                print(f"Email file not found: {email_path}")
                return
                
            email_text = classifier.extract_email_features(email_path)
            if email_text:
                label = classifier.classify([email_text])[0]
                # Print only the classification result, no additional text
                print(label)
                return
            else:
                print(f"Could not extract text from email: {email_path}")
                return
        
        # Multiple emails classification
        if not args.emails:
            print("Either --email or --emails must be provided")
            return
            
        # Get emails to classify
        email_paths = get_emails_from_dir(args.emails)
        if not email_paths:
            print("No emails found. Check your input directory.")
            return
        
        print(f"Found {len(email_paths)} emails to classify")
        
        # Extract features and classify
        results = []
        for path in email_paths:
            email_text = classifier.extract_email_features(path)
            if email_text:
                label = classifier.classify([email_text])[0]
                results.append((path, label))
        
        # Print or save results
        if args.output:
            with open(args.output, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Email', 'Classification'])
                for path, label in results:
                    writer.writerow([path, label])
            print(f"Classification results saved to: {args.output}")
        else:
            print("\nClassification Results:")
            for path, label in results:
                print(f"{path}: {label}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

