import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import requests
import time
import json
from urllib.parse import urljoin
from config.settings import settings

class TDSCourseScraper:
    def __init__(self):
        self.base_url = settings.TDS_COURSE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_markdown_files(self):
        """Get list of markdown files to scrape"""
        markdown_files = [
            "README.md",          # Main course content
            "docker.md",          # Docker section
            "python.md",          # Python section
            "git.md",             # Git section
            "data.md",            # Data section
            "ml.md",              # Machine Learning section
            "projects.md",        # Projects section
            "assignments.md",     # Assignments (if exists)
            "resources.md",       # Resources (if exists)
        ]
        return markdown_files
    
    def scrape_markdown_file(self, filename):
        """Scrape a single markdown file"""
        url = f"{self.base_url}/{filename}"
        
        try:
            print(f"Scraping: {url}")
            time.sleep(settings.REQUEST_DELAY)  # Rate limiting
            
            response = self.session.get(url)
            
            if response.status_code == 200:
                content = response.text
                print(f"✓ Successfully scraped {filename}: {len(content)} characters")
                return content
            else:
                print(f"✗ Failed to scrape {filename}: HTTP {response.status_code}")
                return None
                
        except requests.RequestException as e:
            print(f"✗ Error scraping {filename}: {e}")
            return None
    
    def clean_markdown_content(self, content):
        """Clean and process markdown content"""
        if not content:
            return ""
        
        # Remove excessive whitespace while preserving structure
        lines = content.splitlines()
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped:  # Keep non-empty lines
                cleaned_lines.append(stripped)
            elif cleaned_lines and cleaned_lines[-1]:  # Keep single empty lines
                cleaned_lines.append("")
        
        return '\n'.join(cleaned_lines)
    
    def save_content(self, content, filename):
        """Save scraped content to file"""
        os.makedirs(settings.RAW_DATA_PATH, exist_ok=True)
        filepath = os.path.join(settings.RAW_DATA_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"💾 Saved content to {filepath}")
        return filepath
    
    def scrape_all_content(self):
        """Main method to scrape all TDS course content"""
        print("🚀 Starting TDS Course content scraping...")
        print("📁 Using markdown file approach")
        
        markdown_files = self.get_markdown_files()
        scraped_data = {}
        successful_scrapes = 0
        
        for i, filename in enumerate(markdown_files):
            print(f"\n📄 Processing file {i+1}/{len(markdown_files)}: {filename}")
            
            content = self.scrape_markdown_file(filename)
            
            if content:
                # Clean the content
                cleaned_content = self.clean_markdown_content(content)
                
                # Store in our data structure
                page_name = filename.replace('.md', '')
                scraped_data[page_name] = {
                    'filename': filename,
                    'url': f"{self.base_url}/{filename}",
                    'content': cleaned_content,
                    'raw_length': len(content),
                    'cleaned_length': len(cleaned_content),
                    'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save individual file
                output_filename = f"tds_course_{page_name}.md"
                self.save_content(cleaned_content, output_filename)
                successful_scrapes += 1
            
            # Rate limiting between requests
            time.sleep(settings.REQUEST_DELAY)
        
        # Save combined data as JSON
        combined_file = os.path.join(settings.RAW_DATA_PATH, 'tds_course_all.json')
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Scraping completed!")
        print(f"📊 Successfully scraped {successful_scrapes}/{len(markdown_files)} files")
        print(f"💾 Combined data saved to: {combined_file}")
        
        # Print summary
        total_content = sum(data['cleaned_length'] for data in scraped_data.values())
        print(f"📈 Total content collected: {total_content:,} characters")
        
        return scraped_data

if __name__ == "__main__":
    scraper = TDSCourseScraper()
    try:
        data = scraper.scrape_all_content()
        print(f"\n🎉 Successfully scraped {len(data)} files from TDS course")
        
        # Show what we got
        for page_name, info in data.items():
            print(f"  📄 {page_name}: {info['cleaned_length']:,} characters")
            
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
