from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import random

# Function to introduce a random delay
def random_delay():
    time.sleep(random.uniform(2, 5))  # Random delay between 2 to 5 seconds

# List of User-Agents to rotate
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/602.3.12 (KHTML, like Gecko) Version/10.0.3 Safari/602.3.12'
]

# List of URLs to scrape
urls_to_scrape = [
    'https://www.pcgs.com/coinfacts/category/twenty-cents-quarters/washington-quarter/type-2-clad-1965-1998/1640'
]

# Set up Selenium with headless Chrome and additional options
options = Options()
options.headless = True
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--allow-running-insecure-content')
options.add_argument('--disable-extensions')
options.add_argument('--disable-browser-side-navigation')
options.add_argument('--disable-infobars')

# Initialize WebDriver with WebDriverManager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

for url in urls_to_scrape:
    headers = {
        'User-Agent': random.choice(user_agents),
        'Referer': 'https://www.pcgs.com/',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive'
    }
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": random.choice(user_agents)})
    driver.get(url)
    time.sleep(5)  # Allow time for the page to load
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    images = soup.find_all('img')
    for item in images:
        print(item['src'])
    
    random_delay()

driver.quit()
