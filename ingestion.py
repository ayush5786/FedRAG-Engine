import feedparser
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

TRASH_TITLES = ["commencement", "welcome", "opening remarks", "graduation", "community", "welcoming"]
MACRO_KEYWORDS = ["inflation", "rates", "cpi", "fomc", "labor", "employment", "policy", "yield"]
MIN_KEYWORD_MATCHES = 3

def metadata_filter(title):
    title_lower = title.lower()
    if any(word in title_lower for word in TRASH_TITLES):
        return "DROP"
    return "PASS"

def keyword_bouncer(text):
    text_lower = text.lower()
    match_count = sum(1 for word in MACRO_KEYWORDS if word in text_lower)
    if match_count >= MIN_KEYWORD_MATCHES:
        return True
    return False

def scrape_speech_text(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        date_tag = soup.find('p', class_='article__time')
        if date_tag:
            raw_date = date_tag.get_text(strip=True)
            try:
                # Step 1: Tell Python to read "May 14, 2026" as a real date
                parsed_date = datetime.strptime(raw_date, "%B %d, %Y")
                # Step 2: Tell Python to rewrite it with the Day of the Week ("%A")
                speech_date = parsed_date.strftime("%A, %B %d, %Y")
            except ValueError:
                # Fallback: If the Fed writes a weird format we didn't expect, just print the raw text
                speech_date = raw_date
        else:
            speech_date = "Date not provided"
        paragraphs = soup.find_all('p')
        speech_text = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        return speech_text, speech_date
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "", "Date not provided."

def fetch_speech_list(rss_url, target_speeches=10):
    """Fast RSS metadata fetch. Zero scraping happens here."""
    feed = feedparser.parse(rss_url)
    speech_metadata_list = []
    
    for entry in feed.entries:
        if len(speech_metadata_list) >= target_speeches:
            break
            
        title = entry.title
        link = entry.link
        
        # Stage 1: Fast Title Filter (No network requests)
        if metadata_filter(title) == "DROP":
            continue
            
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            clean_date = time.strftime("%B %d, %Y", entry.published_parsed)
        else:
            clean_date = "Unknown Date"
            
        speech_metadata_list.append({
            "title": title,
            "url": link,
            "date": clean_date
        })
        
    return speech_metadata_list
