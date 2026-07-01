#!/usr/bin/env python3
"""
Chub.ai Smart Proxy - Multi-profile proxy for various AI providers
Handles customer header support, modular endpoint setup, hybrid reasoning consideration
"""

import os
import sys
import json
import yaml
import requests
from flask import Flask, request, Response, jsonify
from pathlib import Path
from datetime import datetime
import logging
from urllib.parse import urlparse

app = Flask(__name__)

# Configuration
PROXY_PORT = 8080
CONFIG_FILE = 'config.yaml'
profiles = {}
default_profile = None
stats = {'requests': 0, 'last_profile': None, 'last_request': None}

# Disable Flask's default logging for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def load_config():
    """Load configuration from YAML file"""
    global profiles, default_profile
    
    config_path = Path(CONFIG_FILE)
    
    # Create default config if it doesn't exist
    if not config_path.exists():
        create_default_config()
        print(f"Created default {CONFIG_FILE} - Please edit it with your API keys!")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        profiles = config.get('profiles', {})
        default_profile = config.get('default_profile', 'openrouter')
        
        # Validate and process environment variables
        for profile_name, profile in profiles.items():
            if 'api_key' in profile:
                # Replace environment variable references
                if profile['api_key'].startswith('${') and profile['api_key'].endswith('}'):
                    env_var = profile['api_key'][2:-1]
                    profile['api_key'] = os.environ.get(env_var, '')
                    if not profile['api_key']:
                        print(f"Warning: {env_var} not set for profile '{profile_name}'")
        
        return True
    except Exception as e:
        print(f"Error loading config: {e}")
        return False

def create_default_config():
    """Create a default configuration file matching config.yaml.example"""
    default_config = {
        'default_profile': 'openrouter',
        'profiles': {
            'openrouter': {
                'name': 'OpenRouter (All Models)',
                'base_url': 'https://openrouter.ai/api/v1',
                'api_key': '${OPENROUTER_API_KEY}',
                'headers': {
                    'HTTP-Referer': 'http://localhost:8080',
                    'X-Title': 'Chub.ai Proxy'
                },
                # OpenRouter handles reasoning automatically for compatible models
                'reasoning': {
                    'enabled': True,
                    'effort': 'high',
                    'exclude': False  # Set to True to hide reasoning from output
                }
            },
            # Additional profiles can be added by copying from config.yaml.example
            # Common ones include: deepseek-direct, moonshot, mistral
        }
    }
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

def get_profile_from_path(path):
    """Extract profile name from URL path"""
    # Remove leading/trailing slashes and split
    clean = path.strip('/')
    if not clean:
        return default_profile, ''
    
    parts = clean.split('/')
    
    # Check if first part is a profile name
    if parts[0] in profiles:
        # Return profile and the rest of the path
        remaining = '/'.join(parts[1:]) if len(parts) > 1 else ''
        return parts[0], remaining
    
    # No profile in path, use default and return full path
    return default_profile, clean

def transform_request(data, profile):
    """Transform request based on profile settings"""
    if not isinstance(data, dict):
        return data
    
    # Handle model forcing
    if 'force_model' in profile:
        data['model'] = profile['force_model']
    
    # Handle model mapping (for DeepSeek direct)
    if 'model_map' in profile and 'model' in data:
        model = data.get('model', '')
        if model in profile['model_map']:
            data['model'] = profile['model_map'][model]
    
    # Handle reasoning parameters - simple passthrough
    # OpenRouter will apply them automatically to compatible models
    if 'reasoning' in profile:
        data['reasoning'] = profile['reasoning']
    
    # Handle max_tokens vs max_completion_tokens
    if 'max_completion_tokens' in data and 'max_tokens' not in data:
        data['max_tokens'] = data.pop('max_completion_tokens')
    
    return data

def make_request(profile, path, method, headers, data, query_string):
    """Make request to the target API"""
    # Build target URL
    base_url = profile.get('base_url', '')
    
    # If base_url already has a path (like /chat/completions), use it as base
    # Otherwise, append the incoming path
    if '/chat/completions' in base_url or '/messages' in base_url:
        # Profile has a specific endpoint, only append additional path if exists
        if path and path != 'chat/completions':
            # Remove chat/completions from path if it's already in base_url
            path = path.replace('chat/completions', '').strip('/')
            if path:
                target_url = f"{base_url.rstrip('/')}/{path}"
            else:
                target_url = base_url
        else:
            target_url = base_url
    else:
        # Profile is just a base API URL, append the full path
        target_url = f"{base_url.rstrip('/')}/{path}" if path else base_url
    
    # Add query string if present
    if query_string:
        target_url += f"?{query_string.decode()}"
    
    # Debug log the final URL
    print(f"[DEBUG] Target URL: {target_url}")
    
    # Prepare headers
    proxy_headers = {}
    
    # Add profile-specific headers
    if 'headers' in profile:
        proxy_headers.update(profile['headers'])
    
    # Add API key
    if 'api_key' in profile and profile['api_key']:
        proxy_headers['Authorization'] = f"Bearer {profile['api_key']}"
    
    # Pass through original headers (skip problematic ones)
    skip_headers = {'host', 'authorization', 'content-length'}
    for key, value in headers:
        if key.lower() not in skip_headers:
            proxy_headers[key] = value
    
    # Make the request
    response = requests.request(
        method=method,
        url=target_url,
        headers=proxy_headers,
        data=data,
        stream=True,
        timeout=300,
        verify=True,
        allow_redirects=False
    )
    
    return response

@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Expose-Headers'] = '*'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD', 'PATCH'])
def proxy_request(path):
    """Main proxy function"""
    
    # Handle preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    # Handle browser hitting profile root (e.g., /openrouter/)
    if request.method == 'GET' and path in profiles and not request.query_string:
        return jsonify({
            "message": f"Profile '{path}' is active",
            "base_url": profiles[path].get('base_url'),
            "status": "ready",
            "hint": "Append /models to list available models"
        }), 200
    
    # Get profile from path
    profile_name, clean_path = get_profile_from_path(path)
    
    # Always show debug for path routing
    print(f"\n[DEBUG] Raw path: '{path}'")
    print(f"[DEBUG] Extracted: profile='{profile_name}', clean_path='{clean_path}'")
    
    if profile_name not in profiles:
        print(f"\nProfile not found: '{profile_name}'")
        print(f"   Raw path: '{path}'")
        print(f"   Available: {list(profiles.keys())}")
        return jsonify({
            "error": f"Profile '{profile_name}' not found. Available profiles: {', '.join(profiles.keys())}",
            "debug": {
                "raw_path": path,
                "extracted_profile": profile_name,
                "extracted_path": clean_path,
                "available_profiles": list(profiles.keys())
            }
        }), 404
    
    profile = profiles[profile_name]
    
    # Update stats
    stats['requests'] += 1
    stats['last_profile'] = profile_name
    stats['last_request'] = datetime.now().strftime('%H:%M:%S')
    
    # Log incoming request
    print(f"\n{'='*60}")
    print(f"[{stats['last_request']}] INCOMING REQUEST")
    print(f"Profile: {profile_name}")
    print(f"Method: {request.method}")
    print(f"Path: {clean_path}")
    if request.query_string:
        print(f"Query: {request.query_string.decode()}")
    
    # Log headers (filter sensitive ones) - commented out for cleaner output
    # print(f"\nHeaders IN:")
    # for key, value in request.headers:
    #     if key.lower() not in ['authorization', 'x-api-key', 'api-key']:
    #         print(f"  {key}: {value}")
    #     else:
    #         print(f"  {key}: [REDACTED]")
    
    # Get request data
    data = None
    original_json_data = None
    transformed_json_data = None
    is_streaming = False
    
    if request.method == 'GET':
        if 'models' in clean_path:
            print(f"\nRequest type: GET → Fetching model list")
        else:
            print(f"\nRequest type: GET → {clean_path or 'root'}")
    elif request.method in ['POST', 'PUT', 'PATCH']:
        data = request.get_data()
        try:
            # Parse JSON data
            original_json_data = json.loads(data) if data else {}
            
            # Check if streaming is enabled
            is_streaming = original_json_data.get('stream', False)
            
            # Log original payload
            print(f"\nPayload IN (original):")
            print(json.dumps(original_json_data, indent=2))
            
            # Transform the request
            transformed_json_data = transform_request(original_json_data.copy(), profile)
            
            # Log transformed payload if different
            if transformed_json_data != original_json_data:
                print(f"\nPayload OUT (transformed):")
                print(json.dumps(transformed_json_data, indent=2))
            else:
                print(f"\nPayload OUT: [unchanged]")
            
            data = json.dumps(transformed_json_data).encode('utf-8')
            
        except json.JSONDecodeError:
            # Not JSON, pass through as-is
            print(f"\nPayload: [non-JSON data, {len(data)} bytes]")
            pass
    
    try:
        # Make the request
        response = make_request(profile, clean_path, request.method, request.headers, data, request.query_string)
        
        # Log response status
        print(f"\n{'─'*60}")
        print(f"RESPONSE:")
        print(f"Status: {response.status_code} {response.reason}")
        
        # Log response headers - commented out for cleaner output
        # print(f"\nHeaders RETURNED:")
        # for key, value in response.headers.items():
        #     if key.lower() not in ['set-cookie']:
        #         print(f"  {key}: {value}")
        
        # Handle response based on streaming
        if is_streaming:
            print(f"\nResponse: [STREAMING ENABLED - content not logged]")
            print(f"{'='*60}\n")
            
            # Stream the response
            def generate():
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=False):
                    if chunk:
                        yield chunk
        else:
            # Non-streaming: capture and log the full response
            response_content = response.content
            
            try:
                response_json = json.loads(response_content)
                print(f"\nResponse BODY:")
                # Limit output for huge responses like model lists
                json_str = json.dumps(response_json, indent=2)
                if len(json_str) > 8000:
                    print(json_str[:8000])
                    print(f"\n... [truncated - {len(json_str)} total characters]")
                else:
                    print(json_str)
                
                # Check for reasoning in response
                if 'choices' in response_json:
                    for choice in response_json.get('choices', []):
                        if 'message' in choice:
                            msg = choice['message']
                            if 'reasoning' in msg or 'reasoning_content' in msg:
                                print(f"\nREASONING DETECTED in response")
                            content = msg.get('content', '')
                            if '<think>' in content or '</think>' in content:
                                print(f"\nTHINKING TAGS in response content")
                
            except json.JSONDecodeError:
                # Not JSON or error
                if len(response_content) < 1000:
                    print(f"\nResponse BODY (non-JSON):")
                    print(response_content.decode('utf-8', errors='ignore'))
                else:
                    print(f"\nResponse BODY: [non-JSON, {len(response_content)} bytes]")
            
            print(f"{'='*60}\n")
            
            # Return the captured content
            def generate():
                yield response_content
        
        # Prepare response headers
        response_headers = []
        skip_response_headers = {'connection', 'transfer-encoding', 'content-encoding'}
        
        for key, value in response.headers.items():
            if key.lower() not in skip_response_headers:
                response_headers.append((key, value))
        
        return Response(
            generate(),
            status=response.status_code,
            headers=response_headers
        )
        
    except requests.exceptions.Timeout:
        print(f"\n{'─'*60}")
        print(f"ERROR: Request timeout for {profile_name}")
        print(f"{'='*60}\n")
        return jsonify({"error": "Request timed out"}), 504
    except requests.exceptions.ConnectionError as e:
        print(f"\n{'─'*60}")
        print(f"ERROR: Connection failed for {profile_name}")
        print(f"Details: {e}")
        print(f"{'='*60}\n")
        return jsonify({"error": "Failed to connect to API"}), 502
    except Exception as e:
        print(f"\n{'─'*60}")
        print(f"ERROR in {profile_name}")
        print(f"Details: {e}")
        print(f"{'='*60}\n")
        return jsonify({"error": str(e)}), 500

def print_startup_message():
    """Print a startup message"""
    print("\n" + "="*60)
    print("Chub.ai Smart Proxy - v1.1")
    print("="*60)
    
    if not profiles:
        print("❌ No profiles loaded. Please check your config.yaml")
        return
    
    print("\nAvailable Endpoints:")
    print("-"*60)
    
    for profile_name, profile in profiles.items():
        name = profile.get('name', profile_name)
        is_default = " (default)" if profile_name == default_profile else ""
        api_key_status = "✅" if profile.get('api_key') else "❌ No API key"
        
        print(f"\n  {name}{is_default}")
        print(f"  URL: http://localhost:{PROXY_PORT}/{profile_name}")
        print(f"  Status: {api_key_status}")
    
    print("\n" + "-"*60)
    print("\nQuick Setup for Chub.ai:")
    print("  1. Copy one of the URLs above")
    print("  2. In a Chub.ai chat, go to Settings → Secrets → OpenAI")
    print("  3. Select 'Reverse Proxy'")
    print("  4. Paste the chosen endpoint URL into 'OAI Reverse Proxy'")
    print("  5. Leave API key blank (it's in config.yaml)")
    print("  6. Click 'Check Proxy` before closing window and refreshing page (F5).")
    
    print("\nLogs:")
    print("-"*60 + "\n")

if __name__ == '__main__':
    print("\nLoading configuration...")
    
    if not load_config():
        print(f"\n⚠️  Please edit {CONFIG_FILE} with your API keys and restart.")
        print("Press Enter to exit...")
        input()
        sys.exit(1)
    
    print_startup_message()
    
    try:
        app.run(host='0.0.0.0', port=PROXY_PORT, debug=True)
    except KeyboardInterrupt:
        print("\n\nProxy stopped. Goodbye!")
    except Exception as e:
        print(f"\nFailed to start proxy: {e}")
        print("Press Enter to exit...")
        input()
