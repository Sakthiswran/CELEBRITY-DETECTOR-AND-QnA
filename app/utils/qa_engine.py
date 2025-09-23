import os
import requests

class QAEngine:

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    ## name comes from the celebrity_detector. We returned name there
    def ask_about_celebrity(self, name, question):
        headers = {
            "Authorization": f"Bearer {self.api_key}", ## Take this API key for authorization purpose
            "Content-Type": "application/json" ## Telling the application server that we are sending a json format
        }

        prompt = f"""
                    You are a AI Assistant that knows a lot about celebrities. You have to answer questions about {name} concisely and accurately.
                    Question : {question}
                    """
        
        # Creating a pull request to send to our API
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}], ## Add user message
            "temperature": 0.5,
            "max_tokens": 512
        }

        response = requests.post(self.api_url, headers = headers, json = payload)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'] # fetch content -> final text output
        
        return "Sorry I couldn't find the answer."