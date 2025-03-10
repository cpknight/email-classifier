#!/usr/bin/env python3
"""
Email Classifier LLM - Classifies emails using LLM providers

This script classifies emails as either 'correspondence' or 'notifications'
using various LLM providers like OpenAI, Anthropic, or Google.

Usage:
    ./email_classifier_llm.py classify --email test_emails/notifications/email5.eml --provider openai
"""

import argparse
import os
import sys
import abc
from email.parser import BytesParser
from email.policy import default
from typing import Dict, Optional, Any

# Import LLM libraries conditionally
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class EmailParser:
    """Parse email files and extract content for classification."""
    
    def __init__(self):
        self.parser = BytesParser(policy=default)
    
    def parse_email(self, email_path):
        """Parse an email file into a usable format."""
        try:
            with open(email_path, 'rb') as f:
                msg = self.parser.parse(f)
            
            # Extract basic info
            subject = msg.get('Subject', '')
            sender = msg.get('From', '')
            
            # Get the body
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/plain':
                        body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                        break
                else:
                    body = "No text content found."
            else:
                body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
            
            return {
                'subject': subject,
                'from': sender,
                'body': body
            }
        
        except Exception as e:
            print(f"Error parsing email: {e}")
            sys.exit(1)


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""
    
    @abc.abstractmethod
    def classify_email(self, email_data: Dict[str, Any]) -> str:
        """Classify an email as 'correspondence' or 'notifications'."""
        pass
    
    @classmethod
    def get_provider(cls, provider_name: str, api_key: Optional[str] = None) -> 'LLMProvider':
        """
        Factory method to get the appropriate LLM provider instance.
        
        Args:
            provider_name: Name of the LLM provider ('openai', 'anthropic', etc.)
            api_key: Optional API key. If not provided, will try to get from environment
            
        Returns:
            LLMProvider: An instance of the appropriate LLM provider
            
        Raises:
            ImportError: If the required module for the provider is not installed
            ValueError: If the provider is not supported or if the API key is missing
        """
        provider_name = provider_name.lower()
        
        # Validate provider name first
        if provider_name not in ['openai', 'anthropic', 'google']:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
            
        # Check for OpenAI provider
        if provider_name == 'openai':
            # Check if the module is imported successfully
            if openai is None:
                raise ImportError("OpenAI Python package not installed. Install with: pip install openai")
            return OpenAIProvider(api_key)
            
        # Check for Anthropic provider
        elif provider_name == 'anthropic':
            # Check if the module is imported successfully
            if anthropic is None:
                raise ImportError("Anthropic Python package not installed. Install with: pip install anthropic")
            return AnthropicProvider(api_key)
            
        # Check for Google provider
        elif provider_name == 'google':
            # Check if the module is imported successfully
            if genai is None:
                raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
            return GoogleProvider(api_key)


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLM provider."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key.")
        
        # Set the API key
        openai.api_key = self.api_key
    
    def classify_email(self, email_data):
        """Classify an email using OpenAI."""
        prompt = f"""
        Classify this email as either 'correspondence' (needs a response) or 'notifications' (no response needed):
        
        Subject: {email_data['subject']}
        From: {email_data['from']}
        
        {email_data['body']}
        
        Classification:
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an email classifier that only responds with either 'correspondence' or 'notifications'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip().lower()
            
            # Normalize result
            if "correspond" in result:
                return "correspondence"
            else:
                return "notifications"
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "correspondence"  # Default in case of error


class AnthropicProvider(LLMProvider):
    """Anthropic implementation of LLM provider."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or use --api-key.")
        
        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def classify_email(self, email_data):
        """Classify an email using Anthropic."""
        prompt = f"""
        Classify this email as either 'correspondence' (needs a response) or 'notifications' (no response needed):
        
        Subject: {email_data['subject']}
        From: {email_data['from']}
        
        {email_data['body']}
        
        Classification:
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                temperature=0.1,
                system="You are an email classifier that only responds with either 'correspondence' or 'notifications'.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.content[0].text.strip().lower()
            
            # Normalize result
            if "correspond" in result:
                return "correspondence"
            else:
                return "notifications"
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return "correspondence"  # Default in case of error


class GoogleProvider(LLMProvider):
    """Google implementation of LLM provider."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable or use --api-key.")
        
        # Configure API
        genai.configure(api_key=self.api_key)
    
    def classify_email(self, email_data):
        """Classify an email using Google."""
        prompt = f"""
        Classify this email as either 'correspondence' (needs a response) or 'notifications' (no response needed):
        
        Subject: {email_data['subject']}
        From: {email_data['from']}
        
        {email_data['body']}
        
        Classification:
        """
        
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            result = response.text.strip().lower()
            
            # Normalize result
            if "correspond" in result:
                return "correspondence"
            else:
                return "notifications"
        except Exception as e:
            print(f"Error calling Google API: {e}")
            return "correspondence"  # Default in case of error


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Email Classifier using LLMs')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify an email')
    classify_parser.add_argument('--email', required=True, help='Path to the email file')
    classify_parser.add_argument('--provider', default='anthropic', choices=['openai', 'anthropic', 'google'],
                                help='LLM provider to use (default: anthropic)')
    classify_parser.add_argument('--api-key', help='API key for the provider')
    
    # Parse args
    args = parser.parse_args()
    
    # Handle classify command
    if args.command == 'classify':
        # Parse the email
        email_parser = EmailParser()
        email_data = email_parser.parse_email(args.email)
        
        # Get the provider
        provider = LLMProvider.get_provider(args.provider, args.api_key)
        
        # Classify the email
        result = provider.classify_email(email_data)
        
        # Output result
        print(result)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
