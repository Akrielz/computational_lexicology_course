import requests as requests
from bs4 import BeautifulSoup

website = "https://ro.wiktionary.org/wiki/Wik%C8%9Bionar:Abrevieri?fbclid=IwAR0OhXNZ9lfshrIlVnofbV1EzoLjhxrUInjGMmDXu5vRr9T-5MEa5XLGFcQ"

# download the website
response = requests.get(website)
# parse the website
soup = BeautifulSoup(response.text, "html.parser")

# parse all abbreviations that are in <li> <b> word <\b> <\li> tags
abbreviations = soup.find_all("li")
# iterate through the abbreviations

abbreviations_list = []
for abbreviation in abbreviations:
    try:
        # get the word
        word = abbreviation.find("b").text

        # check to contain "."
        if "." not in word:
            continue

        # check the rest of the word to only letters
        word_without_dot = word.replace(".", "")
        if not word_without_dot.isalpha():
            continue

        # append the word to the list
        abbreviations_list.append(word)

        print(word)
    except AttributeError:
        pass

# save the list to a file
with open("abbreviations.txt", "w") as f:
    for abbreviation in abbreviations_list:
        f.write(abbreviation + "\n")
