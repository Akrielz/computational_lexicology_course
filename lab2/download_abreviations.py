import requests as requests
from bs4 import BeautifulSoup

website = "https://ro.wiktionary.org/wiki/Wik%C8%9Bionar:Abrevieri?fbclid=IwAR0SCEPxsa_ZLGoDsAMsd6gPM5iKItZSr6WCoMNwU0WJmbhMBQfuAb_n8h8"

# download the website
response = requests.get(website)
# parse the website
soup = BeautifulSoup(response.text, "html.parser")

# parse all abreviations that are in <li> <b> word <\b> <\li> tags
abreviations = soup.find_all("li")