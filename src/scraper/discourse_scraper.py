import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import requests
import time
import json
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from config.settings import settings

class TDSDiscourseScraper:
    def __init__(self):
        # Use main domain for authentication, specific category for content
        self.base_url = "https://discourse.onlinedegree.iitm.ac.in"
        self.category_url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Date range for scraping (Jan 1 - Apr 14, 2025)
        self.start_date = datetime(2025, 1, 1)
        self.end_date = datetime(2025, 4, 14)
        
    def authenticate(self, username, password):
        """Authenticate with Discourse forum"""
        print("🔐 Attempting to authenticate with Discourse...")
        
        # Get the login page to extract CSRF token
        login_url = f"{self.base_url}/login"
        try:
            response = self.session.get(login_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find CSRF token
            csrf_token = None
            csrf_input = soup.find('input', {'name': 'authenticity_token'})
            if csrf_input:
                csrf_token = csrf_input.get('value')
            
            # Login data
            login_data = {
                'login': username,
                'password': password,
                'authenticity_token': csrf_token
            }
            
            # Attempt login
            response = self.session.post(login_url, data=login_data)
            
            if response.status_code == 200 and 'login' not in response.url:
                print("✅ Successfully authenticated with Discourse")
                return True
            else:
                print("❌ Authentication failed")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    def search_tds_topics(self, page=0):
        """Search for TDS-related topics using Discourse API"""
        search_terms = [
            "TDS", "Tools in Data Science", "docker", "python", 
            "git", "data science", "assignment", "project"
        ]
        
        all_topics = []
        
        for term in search_terms:
            print(f"🔍 Searching for: '{term}'")
            
            # Use Discourse search API
            search_url = f"{self.base_url}/search.json"
            params = {
                'q': term,
                'page': page,
                'type_filter': 'topic'
            }
            
            try:
                response = self.session.get(search_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'topics' in data:
                        topics = data['topics']
                        print(f"  📋 Found {len(topics)} topics for '{term}'")
                        
                        # Filter by date range
                        filtered_topics = self.filter_topics_by_date(topics)
                        all_topics.extend(filtered_topics)
                        
                time.sleep(settings.REQUEST_DELAY)
                
            except Exception as e:
                print(f"❌ Error searching for '{term}': {e}")
        
        # Remove duplicates based on topic ID
        unique_topics = {topic['id']: topic for topic in all_topics}.values()
        return list(unique_topics)
    
    def scrape_category_content(self):
        """Scrape content directly from TDS category instead of using search"""
        print(f"\n🔍 Accessing TDS category directly...")
        category_url = f"{self.category_url}.json"
        
        try:
            response = self.session.get(category_url)
            if response.status_code == 200:
                data = response.json()
                
                # Get topics from the category
                topic_list = data.get('topic_list', {})
                topics = topic_list.get('topics', [])
                
                print(f"✅ Found {len(topics)} topics in TDS category")
                
                # Filter by date range and keywords
                filtered_topics = []
                keywords = ['project', 'assignment', 'docker', 'python', 'git', 'tds', 'tools', 'data science']
                
                for topic in topics:
                    # Check date range
                    created_at = topic.get('created_at', '')
                    if created_at:
                        try:
                            topic_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            if self.start_date <= topic_date <= self.end_date:
                                # Check for relevant keywords in title
                                title = topic.get('title', '').lower()
                                if any(keyword in title for keyword in keywords):
                                    filtered_topics.append(topic)
                                    print(f"  📋 Found relevant topic: {topic.get('title', 'Unknown')}")
                        except Exception as e:
                            print(f"⚠️ Date parsing error: {e}")
                
                print(f"📊 Found {len(filtered_topics)} relevant topics in date range")
                return filtered_topics
                
            else:
                print(f"❌ Failed to access category: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error accessing category: {e}")
            return []
    
    def filter_topics_by_date(self, topics):
        """Filter topics by date range (Jan 1 - Apr 14, 2025)"""
        filtered = []
        
        for topic in topics:
            try:
                # Parse topic creation date
                created_at = topic.get('created_at', '')
                if created_at:
                    topic_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    
                    if self.start_date <= topic_date <= self.end_date:
                        filtered.append(topic)
                        
            except Exception as e:
                print(f"⚠️ Date parsing error for topic {topic.get('id', 'unknown')}: {e}")
        
        return filtered
    
    def scrape_topic_content(self, topic_id):
        """Scrape full content of a topic including all posts"""
        print(f"📖 Scraping topic {topic_id}...")
        
        topic_url = f"{self.base_url}/t/{topic_id}.json"
        
        try:
            response = self.session.get(topic_url)
            if response.status_code == 200:
                topic_data = response.json()
                
                # Extract topic information
                topic_info = {
                    'id': topic_id,
                    'title': topic_data.get('title', ''),
                    'created_at': topic_data.get('created_at', ''),
                    'category_id': topic_data.get('category_id', ''),
                    'posts_count': topic_data.get('posts_count', 0),
                    'posts': []
                }
                
                # Extract posts
                if 'post_stream' in topic_data and 'posts' in topic_data['post_stream']:
                    posts = topic_data['post_stream']['posts']
                    
                    for post in posts:
                        post_info = {
                            'id': post.get('id', ''),
                            'username': post.get('username', ''),
                            'created_at': post.get('created_at', ''),
                            'cooked': post.get('cooked', ''),  # HTML content
                            'raw': post.get('raw', ''),       # Raw markdown
                            'post_number': post.get('post_number', 0)
                        }
                        
                        # Clean the content
                        if post_info['cooked']:
                            post_info['cleaned_text'] = self.clean_html_content(post_info['cooked'])
                        elif post_info['raw']:
                            post_info['cleaned_text'] = post_info['raw']
                        else:
                            post_info['cleaned_text'] = ''
                        
                        topic_info['posts'].append(post_info)
                
                print(f"  ✅ Scraped {len(topic_info['posts'])} posts")
                return topic_info
                
            else:
                print(f"  ❌ Failed to scrape topic {topic_id}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"  ❌ Error scraping topic {topic_id}: {e}")
            return None
    
    def clean_html_content(self, html_content):
        """Clean HTML content to extract readable text"""
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
        
        return cleaned_text
    
    def save_discourse_data(self, topics_data, filename):
        """Save scraped Discourse data to file"""
        os.makedirs(settings.RAW_DATA_PATH, exist_ok=True)
        filepath = os.path.join(settings.RAW_DATA_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(topics_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved Discourse data to {filepath}")
        return filepath
    
    def scrape_all_discourse_content(self, username=None, password=None):
        """Main method to scrape all TDS-related Discourse content"""
        print("🚀 Starting TDS Discourse content scraping...")
        print(f"📅 Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        
        # Try to authenticate if credentials provided
        authenticated = False
        if username and password and username != "placeholder_username":
            authenticated = self.authenticate(username, password)
        
        if not authenticated:
            print("⚠️ Proceeding without authentication (public content only)")
            
            # Try to get latest topics from TDS category
            print("\n🔍 Trying to access TDS category topics...")
            latest_url = f"{self.category_url}.json"
            
            try:
                response = self.session.get(latest_url)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Successfully accessed TDS category! Found {len(data.get('topic_list', {}).get('topics', []))} topics")
                    
                    # Get topics from the TDS category
                    all_topics = data.get('topic_list', {}).get('topics', [])
                    
                    # Filter by date range
                    filtered_topics = self.filter_topics_by_date(all_topics)
                    
                    print(f"📊 Found {len(filtered_topics)} TDS topics in date range")
                    
                    if filtered_topics:
                        # Process the found topics
                        scraped_data = {
                            'scraping_info': {
                                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'date_range': {
                                    'start': self.start_date.strftime('%Y-%m-%d'),
                                    'end': self.end_date.strftime('%Y-%m-%d')
                                },
                                'authenticated': False,
                                'total_topics': len(filtered_topics)
                            },
                            'topics': []
                        }
                        
                        successful_scrapes = 0
                        
                        for i, topic in enumerate(filtered_topics[:5]):  # Limit to first 5 for testing
                            print(f"\n📖 Processing topic {i+1}/{min(len(filtered_topics), 5)}: {topic.get('title', 'Unknown')}")
                            
                            topic_content = self.scrape_topic_content(topic['id'])
                            
                            if topic_content:
                                scraped_data['topics'].append(topic_content)
                                successful_scrapes += 1
                            
                            # Rate limiting
                            time.sleep(settings.REQUEST_DELAY)
                        
                        # Save the scraped data
                        filename = f"discourse_tds_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        self.save_discourse_data(scraped_data, filename)
                        
                        print(f"\n✅ Discourse scraping completed!")
                        print(f"📊 Successfully scraped {successful_scrapes}/{min(len(filtered_topics), 5)} topics")
                        
                        # Calculate total content
                        total_posts = sum(len(topic['posts']) for topic in scraped_data['topics'])
                        total_characters = sum(
                            len(post['cleaned_text']) 
                            for topic in scraped_data['topics'] 
                            for post in topic['posts']
                        )
                        
                        print(f"📈 Total posts collected: {total_posts}")
                        print(f"📈 Total content: {total_characters:,} characters")
                        
                        return scraped_data
                    else:
                        print("❌ No TDS topics found in date range")
                        return {}
                else:
                    print(f"❌ Failed to access TDS category: HTTP {response.status_code}")
                    return {}
                    
            except Exception as e:
                print(f"❌ Error accessing TDS category: {e}")
                return {}
        else:
            # Authenticated scraping - use category-based approach instead of search
            topics = self.scrape_category_content()
            
            if not topics:
                print("❌ No relevant topics found in the TDS category")
                return {}
            
            print(f"📊 Processing {len(topics)} relevant topics")
            
            # Process the found topics
            scraped_data = {
                'scraping_info': {
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'date_range': {
                        'start': self.start_date.strftime('%Y-%m-%d'),
                        'end': self.end_date.strftime('%Y-%m-%d')
                    },
                    'authenticated': authenticated,
                    'total_topics': len(topics),
                    'method': 'category_direct_access'
                },
                'topics': []
            }
            
            successful_scrapes = 0
            
            for i, topic in enumerate(topics[:5]):  # Limit to first 5 for testing
                print(f"\n📖 Processing topic {i+1}/{min(len(topics), 5)}: {topic.get('title', 'Unknown')}")
                
                topic_content = self.scrape_topic_content(topic['id'])
                
                if topic_content:
                    scraped_data['topics'].append(topic_content)
                    successful_scrapes += 1
                
                # Rate limiting
                time.sleep(settings.REQUEST_DELAY)
            
            # Save the scraped data
            filename = f"discourse_tds_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_discourse_data(scraped_data, filename)
            
            print(f"\n✅ Discourse scraping completed!")
            print(f"📊 Successfully scraped {successful_scrapes}/{min(len(topics), 5)} topics")
            
            # Calculate total content
            total_posts = sum(len(topic['posts']) for topic in scraped_data['topics'])
            total_characters = sum(
                len(post['cleaned_text']) 
                for topic in scraped_data['topics'] 
                for post in topic['posts']
            )
            
            print(f"📈 Total posts collected: {total_posts}")
            print(f"📈 Total content: {total_characters:,} characters")
            
            return scraped_data

if __name__ == "__main__":
    scraper = TDSDiscourseScraper()
    
    # Get credentials from environment variables
    username = getattr(settings, 'DISCOURSE_USERNAME', 'placeholder_username')
    password = getattr(settings, 'DISCOURSE_PASSWORD', 'placeholder_password')
    
    if username and password and username != "placeholder_username":
        print(f"🔐 Using real credentials for: {username}")
        print("⚠️  Testing with limited scope for safety")
        
        try:
            data = scraper.scrape_all_discourse_content(username, password)
            print(f"\n🎉 Successfully completed Discourse scraping!")
            
        except Exception as e:
            print(f"❌ Error during Discourse scraping: {e}")
    else:
        print("⚠️  No real credentials found, using public content approach")
        try:
            data = scraper.scrape_all_discourse_content()
            print(f"\n🎉 Successfully completed public Discourse scraping!")
        except Exception as e:
            print(f"❌ Error during Discourse scraping: {e}")
