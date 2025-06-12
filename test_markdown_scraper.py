import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import requests
from config.settings import settings

def test_markdown_approach():
    """Test scraping the raw markdown files instead of rendered content"""
    
    # Common TDS markdown file URLs (Docsify typically uses these patterns)
    markdown_urls = [
        "https://tds.s-anand.net/README.md",
        "https://tds.s-anand.net/docker.md", 
        "https://tds.s-anand.net/python.md",
        "https://tds.s-anand.net/git.md",
        "https://tds.s-anand.net/data.md",
        "https://tds.s-anand.net/ml.md",
        "https://tds.s-anand.net/projects.md"
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for url in markdown_urls:
        try:
            print(f"Testing: {url}")
            response = session.get(url)
            
            if response.status_code == 200:
                content = response.text
                print(f"SUCCESS: Found {len(content)} characters")
                print("First 300 characters:")
                print(content[:300])
                print("=" * 50)
                return True
            else:
                print(f"Status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("No markdown files found, trying alternative approach...")
    return False

def test_docsify_config():
    """Check the Docsify configuration to find the actual file structure"""
    try:
        # Try to get the index.html to see how Docsify is configured
        response = requests.get("https://tds.s-anand.net/index.html")
        if response.status_code == 200:
            content = response.text
            print("Found index.html content:")
            print(content[:1000])
            return True
    except Exception as e:
        print(f"Error getting index.html: {e}")
    
    return False

if __name__ == "__main__":
    print("Testing markdown approach...")
    success = test_markdown_approach()
    
    if not success:
        print("\nTesting Docsify configuration...")
        test_docsify_config()
