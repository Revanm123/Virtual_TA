import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import requests
import json
from config.settings import settings

class AIResponder:
    def __init__(self):
        self.aipipe_base_url = "https://api.aipipe.org"
        self.api_key = settings.AIPIPE_TOKEN
        
    def generate_enhanced_response(self, question, context_content, sources):
        """Generate an AI-enhanced response using AIPipe"""
        
        if self.api_key == "your_aipipe_token_here":
            return {
                "answer": context_content[:500] + "...",
                "confidence": 0.8,
                "sources": sources,
                "enhanced": False
            }
        
        prompt = f"""
You are a helpful Teaching Assistant for the Tools in Data Science (TDS) course at IIT Madras. 
A student has asked: "{question}"

Based on the following course materials:
{context_content[:1500]}

Please provide a helpful, accurate response that:
1. Directly answers the student's question
2. Uses information from the course materials
3. Is clear and educational
4. Includes practical examples when relevant
5. Maintains a friendly, supportive tone

Keep the response concise but comprehensive.
"""

        try:
            response = requests.post(
                f"{self.aipipe_base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful TDS course Teaching Assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                ai_response = response.json()
                enhanced_answer = ai_response['choices'][0]['message']['content']
                
                return {
                    "answer": enhanced_answer,
                    "confidence": 0.95,
                    "sources": sources,
                    "enhanced": True
                }
            else:
                return {
                    "answer": context_content[:500] + "...",
                    "confidence": 0.8,
                    "sources": sources,
                    "enhanced": False
                }
                
        except Exception as e:
            return {
                "answer": context_content[:500] + "...",
                "confidence": 0.8,
                "sources": sources,
                "enhanced": False
            }
