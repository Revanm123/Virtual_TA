import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.scraper.course_scraper import TDSCourseScraper

# Test with just one page
scraper = TDSCourseScraper()
try:
    # Test Docker page specifically
    content = scraper.scrape_docsify_content("https://tds.s-anand.net/#/docker")
    
    if content:
        print(f"SUCCESS: Extracted {len(content)} characters")
        print("First 500 characters:")
        print(content[:500])
        print("...")
    else:
        print("FAILED: No content extracted")
        
finally:
    scraper.cleanup()
