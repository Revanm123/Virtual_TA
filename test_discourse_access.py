import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import requests
from config.settings import settings

def test_discourse_access():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Test different endpoints
    endpoints_to_test = [
        "https://discourse.onlinedegree.iitm.ac.in/latest.json",
        "https://discourse.onlinedegree.iitm.ac.in/c/courses.json", 
        "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb.json",
        "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34.json",
        "https://discourse.onlinedegree.iitm.ac.in/categories.json"
    ]
    
    print("Testing Discourse endpoints without authentication:")
    for endpoint in endpoints_to_test:
        try:
            response = session.get(endpoint)
            print(f"  {endpoint}: HTTP {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if 'topic_list' in data:
                    topics = data['topic_list'].get('topics', [])
                    print(f"    Found {len(topics)} topics")
        except Exception as e:
            print(f"  {endpoint}: ERROR - {e}")
    
    print("\nTesting with authentication...")
    # Add authentication logic here if needed

if __name__ == "__main__":
    test_discourse_access()
