import os ## loading our environment variable
## encode your image for API request. 
## Can't send an image directly to an API. Need to encode it first.
## Get an answer from the API. Then decode that image.
import base64 
import requests ## to send the HTTP requests to the APIs

class CelebrityDetector:

    ## Define our constructor
    def __init__(self):
        ## Create some instance variable
        ## 1) GROQ API
        self.api_key = os.getenv("GROQ_API_KEY")
        ## 2) API URL - endpoint for the API to hit
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        ## 3) Pass the model - Vision Transformer
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    ## Methods => Functions
    def identify(self, image_bytes): ## we previously returned the image in bytes format. So we accept in bytes format
        ## Convert image bytes into a base64 encoded string
        ## When working with API, we need the image or anything in base64 format
        encoded_image = base64.b64encode(image_bytes).decode()

        ## Prepare headers for API key and Content Type
        headers = {
            "Authorization": f"Bearer {self.api_key}", ## Take this API key for authorization purpose
            "Content-Type": "application/json" ## Passing the image in the form of JSON
        }

        ## 1) Tell the model which model you want to use
        ## 2) Prompt
        ## 3) Image
        ## One role -> User gives a text. We pass this to LLM
        ## One type
        prompt = {
            "model": self.model, ## llama 4 maverick
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are a celebrity recognition expert AI.
Identify the person in the image. If known, respond in this format:

- **Full Name**:
- **Profession**:
- **Nationality**:
- **Famous For**:
- **Top Achievements**:

If unknown, return "Unknown".                                                     
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}" ## base64 encoded image
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3, ## Higher value -> more creative -> Hallucination. generally 0.2 to 0.5
            "max_tokens": 1024 ## Output
        }

        ## post() method sends a POST request to the specified url
        ## post() method is used when you want to send some data to the server
        response = requests.post(self.api_url, headers = headers, json = prompt)

        ## 200 means success message -> API Request
        if response.status_code == 200:
            # If the request is successful, all the text output from the response is stored in results
            result = response.json()['choices'][0]['message']['content'] ## We fetch the content

            # Extract name from the results
            name = self.extract_name(result)

            return result, name
        
        return "Unknown", ""

    ## If the request is successful, highlight the person's full name
    ## Iterating over the results and fetching the name of the celebrity
    def extract_name(self, content):
        for line in content.splitlines():
            if line.lower().startswith("- **full name**:"):
                return line.split(":")[1].strip()
        
        return "Unknown"