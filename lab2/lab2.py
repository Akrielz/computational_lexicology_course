import re
from typing import List

import rowordnet as wordnet


# Instantiate the WordNet class
wn = wordnet.RoWordNet()


def parse_emails(split_words: List[str]):
    # iterate through the words and find emails
    i = 0
    while i < len(split_words):
        if split_words[i] == "@":
            # remove the @ and the next word
            aron = split_words.pop(i)
            subdomain = split_words.pop(i)
            dot = split_words.pop(i)
            domain = split_words.pop(i)

            # concatenate word at i-1 with the email
            split_words[i-1] = split_words[i-1] + aron + subdomain + dot + domain

        i += 1

    return split_words


def parse_web_url(split_words: List[str]):
    # concatenate the words referearing to a web url into one word

    i = 0
    sign_detected = False
    while i < len(split_words):
        if split_words[i] == "http" or split_words[i] == "https" or split_words[i] == "www":
            while True:
                next_word = split_words[i + 1]
                if next_word in ":/.?=":
                    sign_detected = True
                    # pop the next word
                    split_words.pop(i+1)
                    # concatenate the next word with the current word
                    split_words[i] = split_words[i] + next_word
                elif sign_detected:
                    # pop the next word
                    split_words.pop(i+1)
                    # concatenate the next word with the current word
                    split_words[i] = split_words[i] + next_word
                    sign_detected = False
                else:
                    break

        i += 1

    return split_words


def parse_phone_numbers(split_words: List[str]):
    # concatenate the words referearing to phone number

    i = 0
    while i < len(split_words):
        if (split_words[i][0] in ["+", "("] and split_words[i+1].isdigit()) or split_words[i].isdigit():
            sign_detected = True
            while True:
                next_word = split_words[i + 1]
                if next_word in "()+-/":
                    sign_detected = True
                    # pop the next word
                    split_words.pop(i+1)
                    # concatenate the next word with the current word
                    split_words[i] = split_words[i] + next_word
                elif sign_detected and next_word.isdigit():
                    # pop the next word
                    split_words.pop(i+1)
                    # concatenate the next word with the current word
                    split_words[i] = split_words[i] + next_word
                    sign_detected = False
                else:
                    break

        i += 1

    return split_words


def parse_foreign(split_words: List[str]):
    # read foreign words from foreign.txt
    with open("foreign.txt", "r") as f:
        foreign_words = f.readlines()
        for i in range(len(foreign_words)):
            foreign_words[i] = foreign_words[i].strip()

    i = 0
    while i < len(split_words) - 1:
        current_word = split_words[i]
        next_word = split_words[i+1]
        compound_word = current_word + " " + next_word
        if compound_word.lower() in foreign_words:
            split_words[i] = compound_word
            split_words.pop(i+1)

        i += 1

    return split_words


def parse_ip(split_words: List[str]):
    # iterate through the words and find emails
    i = 0
    while i < len(split_words) - 6:

        current_word = split_words[i]
        # check if current_word is a number
        # an ip is in the next form
        # number . number . number . number

        if current_word.isdigit() and split_words[i+1] == "." and split_words[i+2].isdigit() and split_words[i+3] == "." \
                and split_words[i+4].isdigit() and split_words[i+5] == "." and split_words[i+6].isdigit():

            dot1 = split_words.pop(i+1)
            number2 = split_words.pop(i+1)
            dot2 = split_words.pop(i+1)
            number3 = split_words.pop(i+1)
            dot3 = split_words.pop(i+1)
            number4 = split_words.pop(i+1)

            # concatenate word at i-1 with the email
            split_words[i] = split_words[i] + dot1 + number2 + dot2 + number3 + dot3 + number4

        i += 1

    return split_words


def parse_abbreviations_names(split_words: List[str]):
    # read abbreviations
    with open("abbreviations_names.txt", "r") as f:
        abbreviations = f.read()

    # split abrevaitions in lines
    abbreviations = abbreviations.split("\n")

    i = 0
    while i < len(split_words) - 1:
        current_word = split_words[i]
        next_word = split_words[i+1]
        composed_word = current_word + next_word

        if composed_word.lower() in abbreviations:
            whole_name = gather_compound_name(i+2, split_words)

            split_words[i] = composed_word + " " + whole_name
            split_words.pop(i+1)

        i += 1

    return split_words


def parse_abbreviations_singular(split_words: List[str]):
    # read abbreviations
    with open("abbreviations_singular.txt", "r") as f:
        abbreviations = f.readlines()
        for i in range(len(abbreviations)):
            abbreviations[i] = abbreviations[i].strip()

    i = 1
    while i < len(split_words):
        if split_words[i] == ".":
            sign_detected = True
            j = i + 1
            compound_word = split_words[i-1] + split_words[i]
            len_last_word = 0
            while True:
                if j >= len(split_words):
                    break

                next_word = split_words[j]
                if next_word in ".":
                    sign_detected = True
                    # concatenate the next word with the current word
                    compound_word += next_word
                elif sign_detected:
                    # concatenate the next word with the current word
                    compound_word += next_word
                    len_last_word = len(next_word)
                    sign_detected = False
                else:
                    break

                j += 1

            # remove the last word
            compound_word = compound_word[:-len_last_word]

            # check if the compound word is in the abbreviations list
            if compound_word.lower() in abbreviations:
                split_words[i-1] = compound_word

                # remove the last word
                for k in range(j - i - 1):
                    split_words.pop(i)

        i += 1

    return split_words


def parse_compound_names(split_words: List[str]):
    i = 0
    while i < len(split_words):
        compound_name = gather_compound_name(i, split_words)
        if len(compound_name) > 0:
            split_words.insert(i, compound_name)

        i += 1

    return split_words


def parse_hyphen(split_words: List[str]):
    i = 1
    while i < len(split_words) - 1:
        previous_word = split_words[i-1]
        current_word = split_words[i]
        next_word = split_words[i+1]

        if current_word == "-" and (next_word == "ul" or not(len(previous_word) < 3 or len(next_word) < 3)):
            # remove the hyphen
            split_words.pop(i)
            split_words.pop(i)
            # concatenate the next word with the current word
            split_words[i-1] += current_word + next_word

        i += 1

    return split_words


def parse_end_of_sentence(split_words: List[str]):
    i = 0
    while i < len(split_words):
        current_word = split_words[i]

        if "-" in current_word:
            current_word = current_word.replace("-", "")

            if check_if_word_exists(current_word.lower()):
                split_words[i] = current_word

        i += 1

    return split_words


def gather_compound_name(i, split_words):
    still_name = True
    whole_name = ""
    while still_name:
        if i >= len(split_words):
            if len(whole_name) > 0:
                whole_name = whole_name[:-1]
            break

        name = split_words[i]

        # check if name starts with capital
        if not (name[0].isupper() or name == "-") or len(name) < 3:
            if len(whole_name) > 0:
                whole_name = whole_name[:-1]
            break

        split_words.pop(i)

        if name == "-":
            whole_name = whole_name[:-1]
            whole_name += name
        else:
            whole_name += name + " "
    return whole_name


def check_if_word_exists(word):
    # check with wordnet if the word exists

    synset_ids = wn.synsets(literal=word)
    return len(synset_ids) > 0


def tokenize(text: str):
    # split
    words = re.findall(r"[\w']+|[\-.,!?;@:/()\]\[{}'\"]", text)

    print(words)
    words = parse_emails(words)
    words = parse_web_url(words)
    words = parse_phone_numbers(words)
    words = parse_ip(words)
    words = parse_abbreviations_names(words)
    words = parse_abbreviations_singular(words)
    words = parse_compound_names(words)
    words = parse_hyphen(words)
    words = parse_end_of_sentence(words)
    words = parse_foreign(words)

    print(words)

    return words


def main():
    # read "input.txt"
    with open("input_small.txt", "r") as f:
        data = f.read()

    # tokenize
    words = tokenize(data)


if __name__ == "__main__":
    main()