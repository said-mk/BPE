# this script scrapes all of Paul Graham's essays from his website and saves them to a text file for tokenizer training purposes.

import requests
from bs4 import BeautifulSoup
import time

BASE_URL = 'https://paulgraham.com/'
INDEX_URL = f'{BASE_URL}articles.html'

def main():
    print(f"Fetching index: {INDEX_URL}")
    response = requests.get(INDEX_URL)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # PG's links are typically <a href="essayname.html">
    links = []
    for a in soup.find_all('a'):
        href = a.get('href')
        # Filter out external links, non-html files, and the index itself
        if href and href.endswith('.html') and 'http' not in href and href != 'articles.html':
            if href not in links: # Keep order, avoid duplicates
                links.append(href)
                
    print(f"Found {len(links)} essays. Commencing scrape...")

    with open('..\test\pg_essays.txt', 'w', encoding='utf-8') as f:
        for i, link in enumerate(links):
            url = f"{BASE_URL}{link}"
            print(f"[{i+1}/{len(links)}] Fetching {url}...")
            
            try:
                res = requests.get(url)
                res.raise_for_status()
                page_soup = BeautifulSoup(res.text, 'html.parser')
                
                # Extract text, separate blocks with newlines
                text = page_soup.get_text(separator='\n', strip=True)
                
                # Pro-tip: Inject standard document boundaries for your tokenizer
                f.write(text + "\n\n<|endoftext|>\n\n")
                
                # Do not remove this. Respect the server.
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")

    print("Done. Saved to pg_essays.txt")

if __name__ == '__main__':
    main()