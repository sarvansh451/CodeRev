from bs4 import BeautifulSoup
import requests

# Pretend to be a browser (important for some sites)
headers = {
    'User-Agent': 'Mozilla/5.0'
}

# Fetch the page
response = requests.get('http://books.toscrape.com/', headers=headers)

# Check if request worked
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all book titles inside <h3> tags (the site uses <h3> not <h2>)
    # Each <h3> has an <a> tag with the title in the 'title' attribute
    books = soup.find_all('h3')
    
    for book in books:
        title = book.find('a')['title']
        print(title)
else:
    print("Failed to load page")
