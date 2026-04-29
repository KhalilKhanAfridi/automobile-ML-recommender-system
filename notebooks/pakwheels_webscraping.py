from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

s = Service('C:/Users/Pc/OneDrive/Desktop/chromedriver.exe')
driver = webdriver.Chrome(service=s, options=chrome_options)

driver.get('https://www.pakwheels.com/used-cars/search/-/')
time.sleep(3)

# Click KPK filter
driver.find_element(by=By.XPATH, value='//*[@id="collapse_4"]/div/ul/li[3]/label/a/input').click()
time.sleep(5)

all_html = ""

while True:
    time.sleep(3)

    # Save each page HTML
    all_html += driver.page_source + "\n\n<!-- NEXT PAGE -->\n\n"

    try:
        next_button = driver.find_element(by=By.XPATH,value= "//a[contains(text(),'Next')]")
        driver.execute_script("arguments[0].click();", next_button)
        print("Moving to next page...")
        time.sleep(5)

    except NoSuchElementException:
        print("No more pages")
        break

with open('pakwheels.html', 'w', encoding='utf-8') as f:
    f.write(all_html)
