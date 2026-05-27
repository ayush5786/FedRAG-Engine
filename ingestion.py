import feedparser
import requests
from bs4 import BeautifulSoup

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
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def fetch_valid_speeches(rss_url, max_speeches_to_check=5):
    feed = feedparser.parse(rss_url)
    valid_speeches = []
    
    for entry in feed.entries[:max_speeches_to_check]:
        title = entry.title
        link = entry.link
        
        if metadata_filter(title) == "DROP":
            continue
            
        speech_text = scrape_speech_text(link)
        
        if keyword_bouncer(speech_text):
            valid_speeches.append({
                "title": title,
                "url": link,
                "text": speech_text
            })
            
    return valid_speeches
