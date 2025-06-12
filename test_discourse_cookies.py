import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import requests
import json
from datetime import datetime
from urllib.parse import quote, unquote

def test_discourse_with_real_data():
    """Test Discourse access with properly encoded cookies"""
    
    # Your actual extracted cookies - properly handle the encoding
    cookies = {
        '_ga': 'GA1.1.345206845.1699023158',
        '_ga_QHXRKWW9HH': 'GS1.3.1699595950.3.1.1699598169.0.0.0',
        '_ga_M54BYECMEL': 'GS1.3.1700843878.1.1.1700844115.0.0.0',
        '_ga_WGM85Z9ZZV': 'GS1.1.1707222369.3.1.1707222697.0.0.0',
        '_ga_5HTJMW67XK': 'GS2.1.s1749179307$o87$g0$t1749179345$j22$l0$h0',
        '_fbp': 'fb.2.1707041746489.156896196',
        '_ga_08NPRH5L4M': 'GS2.1.s1749179306$o99$g1$t1749179424$j24$l0$h0',
        '_gcl_aw': 'GCL.1745735304.Cj0KCQjwiLLABhCEARIsAJYS6umb2rHmUw5kqq8lx3QHiIl8F2VDglYYT7wmVy37fiJckzAAxEvYJhMaAub4EALw_wcB',
        # Fix the encoding issue by removing the problematic character
        '_t': 'nIFOOuycTmxm3mDgT2c3LfC00C16NQMEW1O6OBn96QtMVK7PZs1dIFhKhvG74v%2BqvHK7NBlnJh4m%2B9mb8WQ269mopGq9gbjEZaOjhJTPq1w2BNFA%3D--YShgK7sCTBbV2ODm--%2FxaMEvT7OMpcVJyE0xnmzg%3D%3D',
        '_forum_session': '1XZS8jw9Q%2FP0qCRhJu6SFRdgfOlJDnEnGYQAEmTaxzbneWb%2FhBRzRDa0d9Xdsm5OqV3tI%2BKDZvAcFC9CdQ%2BPvgtEmcl%2BlYA1eU5lc8%2FF4VIqk%2B3kgQtXAZt64vWovZnoYB5eeW1lxsyQl%2BClS1UycAvCqB9f1c%2FcvTecm3XqlloRDplGguM9A7HAFZ7s6BOQrCJlQ7e8itRmH6dx9RgpQzGDR6XOZct%2B%2BqDtuuPHsxtTvloKmHTRuuFlHOsupx3gZykpuzQvyykKUGrGx28ws1YDpKnuMQ%3D%3D--KwrJ88cgxNTBKarJ--DmFNkmc2wKniSgHEqPu2sg%3D%3D'
    }
    
    # Use simpler headers to avoid encoding issues
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0',
        'Accept': 'application/json, text/html, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    # The actual API endpoint
    api_url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"
    
    try:
        print(f"🔍 Testing real Discourse API endpoint: {api_url}")
        
        # Create a session with proper encoding
        session = requests.Session()
        session.headers.update(headers)
        
        # Make the request with proper encoding handling
        response = session.get(api_url, cookies=cookies, timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response content type: {response.headers.get('content-type', 'unknown')}")
        
        if response.status_code == 200:
            # Check if it's JSON or HTML
            content_type = response.headers.get('content-type', '')
            
            if 'application/json' in content_type:
                # It's JSON - parse it
                try:
                    data = response.json()
                    print(f"✅ SUCCESS! Got JSON response")
                    
                    # Look for topic list
                    if 'topic_list' in data:
                        topics = data['topic_list'].get('topics', [])
                        print(f"📋 Found {len(topics)} topics in TDS category")
                        
                        # Show first few topics
                        print("\n📝 First 5 topics:")
                        for i, topic in enumerate(topics[:5]):
                            title = topic.get('title', 'No title')
                            created_at = topic.get('created_at', 'Unknown date')
                            print(f"  {i+1}. {title}")
                            print(f"     Created: {created_at}")
                        
                        return data
                    else:
                        print(f"⚠️ Got JSON but no topic_list found")
                        print(f"📊 Response keys: {list(data.keys())}")
                        return data
                        
                except json.JSONDecodeError:
                    print(f"❌ Response is not valid JSON")
                    print(f"📄 Response text (first 500 chars): {response.text[:500]}")
                    return None
            else:
                # It's HTML - we're getting the webpage, not the API
                print(f"⚠️ Got HTML response instead of JSON")
                print(f"📄 This means we're accessing the webpage, not the API")
                
                # Try adding .json to the URL
                json_url = api_url + ".json"
                print(f"🔄 Trying with .json extension: {json_url}")
                
                response2 = session.get(json_url, cookies=cookies, timeout=30)
                print(f"📊 JSON URL Response status: {response2.status_code}")
                
                if response2.status_code == 200:
                    try:
                        data = response2.json()
                        print(f"✅ SUCCESS with .json URL!")
                        
                        if 'topic_list' in data:
                            topics = data['topic_list'].get('topics', [])
                            print(f"📋 Found {len(topics)} topics in TDS category")
                            return data
                        
                    except json.JSONDecodeError:
                        print(f"❌ Still not valid JSON")
                        return None
                
                return None
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"📄 Response: {response.text[:200]}...")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Testing Discourse access with fixed encoding...")
    result = test_discourse_with_real_data()
    
    if result:
        print("\n🎉 SUCCESS! We can access TDS Discourse content!")
        print("✅ Ready to integrate into the main scraper")
    else:
        print("\n❌ Failed to access Discourse content")
        print("💡 Let's try a different approach or proceed with TDS course content only")
