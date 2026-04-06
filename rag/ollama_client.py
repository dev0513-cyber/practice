import json
import urllib.request
import urllib.error

def generate_ollama_response(prompt: str, model: str = "qwen:1.8b") -> str:
    """
    Connects to local Ollama HTTP API and generates a response.
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', '')
    except urllib.error.URLError as e:
        return f"Error connecting to Ollama: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
