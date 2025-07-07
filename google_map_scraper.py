import time
import os
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

class GoogleMapScraper:
    def __init__(self):
        self.output_file_name = "reviews.csv"
        self.headless = False
        self.driver = None
        self.unique_check = []

    def config_driver(self):
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
            options.add_argument('--ignore-ssl-errors=yes')
            options.add_argument('--ignore-certificate-errors')
        options.add_argument("--lang=en-GB")
        options.add_argument("--remote-debugging-port=9222")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def load_companies(self, url):
        print("Getting business info", url)
        self.driver.get(url)
        time.sleep(5)  # Ensure the page is fully loaded
        panel_xpath = "//div[contains(@class, 'm6QErb') and contains(@class, 'DxyBCb') and contains(@class, 'kA9KIf') and contains(@class, 'dS8AEf')]"
        try:
            scrollable_div = self.driver.find_element(By.XPATH, panel_xpath)
        except NoSuchElementException:
            print("Scrollable panel not found.")
            return
        
        flag = True
        i = 0
        while flag:
            print(f"Scrolling to page {i + 2}")
            self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollTop + 6500', scrollable_div)
            time.sleep(2)  # Wait for content to load

            if "You've reached the end of the list." in self.driver.page_source:
                flag = False

            self.get_business_info()
            i += 1

    def get_business_info(self):
        WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'jJc9Ad')))
        try:
            for business in self.driver.find_elements(By.CLASS_NAME, 'jJc9Ad'):
                name = business.find_element(By.CLASS_NAME, 'd4r55').text
                
                # Extract review text
                try:
                    review = business.find_element(By.CLASS_NAME, 'wiI7pd').text
                except NoSuchElementException:
                    review = "No review found"

                # Extract date
                date_elements = business.find_elements(By.CLASS_NAME, 'rsqaWe')
                if date_elements:
                    date_text = date_elements[0].text
                    date_parts = date_text.split(", ")
                    date = date_parts[1] if len(date_parts) > 1 else date_text
                else:
                    date = "No date found"

                # Generate a unique ID to avoid duplicates
                unique_id = "".join([name, review, date])
                if unique_id not in self.unique_check:
                    data = [name, review, date]
                    self.save_data(data)
                    self.unique_check.append(unique_id)
                    print(unique_id)

        except NoSuchElementException as e:
            print(f"An error occurred: {e}")

    def save_data(self, data):
        header = ['ID', 'Client_Name', 'Reviews', 'Date']
        file_exists = os.path.isfile(self.output_file_name)
        with open(self.output_file_name, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([len(self.unique_check)] + data)

# Instantiate the scraper and run
url = "https://www.google.com/maps/place/Girija+Restaurant/@19.3138718,84.7846528,17z/data=!4m12!1m2!2m1!1sGirija+Restaurant!3m8!1s0x3a3d500e5361f391:0x33b13643d297a673!8m2!3d19.3112516!4d84.7851087!9m1!1b1!15sChFHaXJpamEgUmVzdGF1cmFudFoTIhFnaXJpamEgcmVzdGF1cmFudJIBEWluZGlhbl9yZXN0YXVyYW504AEA!16s%2Fg%2F124ss4m2p?authuser=0&entry=ttu&g_ep=EgoyMDI0MTAyNy4wIKXMDSoASAFQAw%3D%3D"
business_scraper = GoogleMapScraper()
business_scraper.config_driver()
business_scraper.load_companies(url)