#!/usr/bin/env python3
"""
Email Responder LLM

This script generates draft responses to emails using a modular LLM interface.
It parses .eml files, extracts relevant information, and uses an LLM to generate
an appropriate response.

Usage:
    # Generate a response to a single email using a specific LLM provider
    python email_responder_llm.py --email path/to/email.eml --provider openai --api-key YOUR_API_KEY

    # Generate a response using environment variables for API keys
    python email_responder_llm.py --email path/to/email.eml --provider anthropic

    # Generate a response with a specific language style or tone
    python email_responder_llm.py --email path/to/email.eml --style professional --provider openai
"""

import os
import re
import json
import argparse
import abc
from typing import Dict, List, Optional, Any
from email.parser import BytesParser
from email.policy import default
from email.header import decode_header
from email.utils import parseaddr

# Import LLM-specific libraries conditionally to avoid unnecessary dependencies
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
    """
    A class to parse email files (.eml) and extract relevant information.
    """
    
    def __init__(self):
        """Initialize the email parser."""
        self.parser = BytesParser(policy=default)
    
    def parse_email(self, email_path: str) -> Dict[str, Any]:
        """
        Parse an email file and extract relevant information.
        
        Args:
            email_path: Path to the email file
            
        Returns:
            dict: Dictionary containing parsed email information
        """
        try:
            with open(email_path, 'rb') as fp:
                msg = self.parser.parse(fp)
            
            # Extract basic headers
            subject = self._decode_header(msg.get('subject', ''))
            from_name, from_email = self._parse_address(msg.get('from', ''))
            to_name, to_email = self._parse_address(msg.get('to', ''))
            cc_list = self._parse_cc(msg.get('cc', ''))
            date = msg.get('date', '')
            message_id = msg.get('message-id', '')
            
            # Extract the body
            body = self._extract_body(msg)
            
            # Create a parsed email object
            email_data = {
                'subject': subject,
                'from': {
                    'name': from_name,
                    'email': from_email
                },
                'to': {
                    'name': to_name,
                    'email': to_email
                },
                'cc': cc_list,
                'date': date,
                'message_id': message_id,
                'body': body,
                'full_text': f"From: {from_name} <{from_email}>\nTo: {to_name} <{to_email}>\nSubject: {subject}\nDate: {date}\n\n{body}"
            }
            
            return email_data
            
        except Exception as e:
            print(f"Error processing email {email_path}: {e}")
            return {}
    
    def _decode_header(self, header: str) -> str:
        """
        Decode email header to handle non-ASCII characters.
        
        Args:
            header: Email header string
            
        Returns:
            str: Decoded header string
        """
        if not header:
            return ""
            
        decoded_parts = []
        for part, encoding in decode_header(header):
            if isinstance(part, bytes):
                if encoding:
                    try:
                        decoded_parts.append(part.decode(encoding))
                    except:
                        decoded_parts.append(part.decode('utf-8', errors='ignore'))
                else:
                    decoded_parts.append(part.decode('utf-8', errors='ignore'))
            else:
                decoded_parts.append(part)
                
        return ''.join(decoded_parts)
    
    def _parse_address(self, address: str) -> tuple:
        """
        Parse an email address into name and email components.
        
        Args:
            address: Email address string, possibly with a display name
            
        Returns:
            tuple: (name, email) parsed from the address
        """
        name, email = parseaddr(address)
        name = self._decode_header(name)
        return name, email
    
    def _parse_cc(self, cc_header: str) -> List[Dict[str, str]]:
        """
        Parse CC header into a list of recipients.
        
        Args:
            cc_header: CC header string
            
        Returns:
            list: List of dictionaries containing name and email for each CC recipient
        """
        if not cc_header:
            return []
            
        cc_list = []
        for cc_part in cc_header.split(','):
            name, email = self._parse_address(cc_part.strip())
            if email:
                cc_list.append({
                    'name': name,
                    'email': email
                })
                
        return cc_list
    
    def _extract_body(self, msg) -> str:
        """
        Extract the body text from an email message.
        
        Args:
            msg: Email message object
            
        Returns:
            str: Body text from the email
        """
        body = ""
        
        if msg.is_multipart():
            # Handle multipart messages
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                # Skip attachments
                if 'attachment' in content_disposition:
                    continue
                    
                # Get the text content
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            body += payload.decode(charset, errors='ignore')
                        except:
                            body += payload.decode('utf-8', errors='ignore')
        else:
            # Handle simple messages
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    body = payload.decode(charset, errors='ignore')
                except:
                    body = payload.decode('utf-8', errors='ignore')
        
        # Clean up the body (remove excessive whitespace, etc.)
        body = re.sub(r'\r\n', '\n', body)
        body = re.sub(r'\n\s*\n\s*\n+', '\n\n', body)
        
        return body.strip()


class LLMProvider(abc.ABC):
    """
    Abstract base class for LLM providers.
    This defines the interface that all LLM provider implementations must follow.
    """
    
    @abc.abstractmethod
    def generate_response(self, email_data: Dict[str, Any], style: str = 'standard') -> str:
        """
        Generate a response to an email using the LLM provider.
        
        Args:
            email_data: Dictionary containing parsed email data
            style: Style/tone for the response (e.g., 'professional', 'friendly')
            
        Returns:
            str: Generated response text
        """
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
            if 'openai' not in globals() or openai is None:
                raise ImportError("OpenAI Python package not installed. Install with: pip install openai")
            return OpenAIProvider(api_key)
            
        # Check for Anthropic provider
        elif provider_name == 'anthropic':
            # Check if the module is imported successfully
            if 'anthropic' not in globals() or anthropic is None:
                raise ImportError("Anthropic Python package not installed. Install with: pip install anthropic")
            return AnthropicProvider(api_key, model='claude-3-haiku-20240307')
            
        # Check for Google provider
        elif provider_name == 'google':
            # Check if the module is imported successfully
            if 'genai' not in globals() or genai is None:
                raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
            return GoogleProvider(api_key)


class OpenAIProvider(LLMProvider):
    """
    OpenAI API implementation of the LLM provider.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment
        """
        if openai is None:
            raise ImportError("OpenAI Python package not installed. Install with: pip install openai")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
    
    def generate_response(self, email_data: Dict[str, Any], style: str = 'standard') -> str:
        """
        Generate a response using OpenAI's API.
        
        Args:
            email_data: Dictionary containing parsed email data
            style: Style/tone for the response
            
        Returns:
            str: Generated response text
        """
        prompt = self._create_prompt(email_data, style)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4",  # Can be configured or passed as parameter
                messages=[
                    {"role": "system", "content": f"You are an email assistant helping to draft a {style} response to an email. Include an appropriate salutation, response text, and closing with signature."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response with OpenAI: {e}")
            return f"Error generating response: {e}"
    
    def _create_prompt(self, email_data: Dict[str, Any], style: str) -> str:
        """
        Create a prompt for the OpenAI API based on the email data.
        
        Args:
            email_data: Dictionary containing parsed email data
            style: Style/tone for the response
            
        Returns:
            str: Prompt for the API
        """
        from_name = email_data['from']['name'] or email_data['from']['email']
        
        prompt = f"""
Please draft a {style} response to the following email:

FROM: {from_name} <{email_data['from']['email']}>
SUBJECT: {email_data['subject']}

{email_data['body']}

Draft a complete response including:
1. An appropriate salutation
2. Response content that addresses the questions or points raised in the email
3. A professional closing with a name

Make the response clear, concise, and helpful.
"""
        return prompt


class AnthropicProvider(LLMProvider):
    """
    Anthropic API implementation of the LLM provider.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'claude-3-haiku-20240307'):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment
            model: Anthropic model to use, defaults to 'claude-3-haiku-20240307'
        """
        if anthropic is None:
            raise ImportError("Anthropic Python package not installed. Install with: pip install anthropic")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        # Store model parameter
        self.model = model
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate_response(self, email_data: Dict[str, Any], style: str = 'standard') -> str:
        """
        Generate a response using Anthropic's API.
        
        Args:
            email_data: Dictionary containing parsed email data
            style: Style/tone for the response
            
        Returns:
            str: Generated response text
        """
        messages = self._create_prompt(email_data, style)
        
        try:
            system_message = f"You are drafting an email response on my behalf - my name is Chris (Knight) and I'm kind of pedantic in style. Please take a personable, professional (unless its clearly a personal message), and (for emphasis) {style} response to the email. If the sender's request is unclear or requires further explanation, please ask. Include an appropriate salutation, response text, and appropriate closing salutation with my first name (and last name if it is a particularily formal email). I personally will be reviewing the draft email before it is sent."
            
            response = self.client.messages.create(
                system=system_message,
                messages=messages,
                model=self.model,
                max_tokens=800,
                temperature=0.7
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"Error generating response with Anthropic: {e}")
            return f"Error generating response: {e}"
    
    def _create_prompt(self, email_data: Dict[str, Any], style: str) -> List[Dict[str, str]]:
        """
        Create message structure for the Anthropic API based on the email data.
        
        Args:
            email_data: Dictionary containing parsed email data
            style: Style/tone for the response
            
        Returns:
            List[Dict[str, str]]: Messages for the API
        """
        from_name = email_data['from']['name'] or email_data['from']['email']
        
        user_content = f"""
I need help drafting a {style} response to an email. Here's the email I received:

FROM: {from_name} <{email_data['from']['email']}>
SUBJECT: {email_data['subject']}

{email_data['body']}

Please draft a complete response including:
1. An appropriate salutation
2. Response content that addresses the questions or points raised in the email
3. A professional closing with a name

Make the response clear, concise, and helpful in a {style} tone. Only provide the text content of the email response, without any additional explanations.
"""

        return [
            {"role": "user", "content": user_content}
        ]


class GoogleProvider(LLMProvider):
    """
    Google AI implementation of the LLM provider.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google provider.
        
        Args:
            api_key: Google API key. If None, will try to get from environment
        """
        if genai is None:
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not provided and not found in environment variables")
        
        # Initialize Google AI client
        genai.configure(api_key=self.api_key)
    
    def generate_response(self, email_data: Dict[str, Any], style: str = 'standard') -> str:
        """
        Generate a response using Google's generative AI API.
        
        Args:
            email_data: Dictionary containing parsed email data
            style: Style/tone for the response
            
        Returns:
            str: Generated response text
        """
        prompt = self._create_prompt(email_data, style)
        
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating response with Google AI: {e}")
            return f"Error generating response: {e}"
    
    def _create_prompt(self, email_data: Dict[str, Any], style: str) -> str:
        """
        Create a prompt for the Google AI API based on the email data.
        
        Args:
            email_data: Dictionary containing parsed email data
            style: Style/tone for the response
            
        Returns:
            str: Prompt for the API
        """
        from_name = email_data['from']['name'] or email_data['from']['email']
        
        prompt = f"""
You are an email assistant helping to draft a {style} response to an email.

Original email:
FROM: {from_name} <{email_data['from']['email']}>
SUBJECT: {email_data['subject']}

{email_data['body']}

Please draft a complete response including:
1. An appropriate salutation
2. Response content that addresses the questions or points raised in the email
3. A professional closing with a name

Make the response clear, concise, and helpful in a {style} tone.
Only provide the text content of the email response, without any additional explanations.
"""
        return prompt


def main():
    """
    Main function to parse arguments and generate email response.
    """
    parser = argparse.ArgumentParser(description='Generate a response to an email using an LLM.')
    parser.add_argument('--email', required=True, help='Path to the .eml file')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic', 'google'], 
                       help='LLM provider to use')
    parser.add_argument('--api-key', help='API key for the LLM provider (optional, can use env var)')
    parser.add_argument('--style', default='standard', 
                       choices=['standard', 'professional', 'friendly', 'formal', 'casual'],
                       help='Style/tone of the response')
    args = parser.parse_args()
    
    # Parse the email
    email_parser = EmailParser()
    email_data = email_parser.parse_email(args.email)
    
    if not email_data:
        print("Error: Failed to parse email.")
        return 1
    
    try:
        # Get the appropriate LLM provider
        provider = LLMProvider.get_provider(args.provider, args.api_key)
        
        # Generate the response
        response = provider.generate_response(email_data, args.style)
        
        # Output the response to STDOUT
        print(response)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

