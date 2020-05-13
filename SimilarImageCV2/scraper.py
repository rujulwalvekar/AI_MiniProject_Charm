from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time

base_url="https://www.thesouledstore.com/explore/t-shirts"
driver=webdriver.Chrome('/home/ritom/Downloads/chromedriver')

driver.get(base_url)
time.sleep(10)

print("Hullo")