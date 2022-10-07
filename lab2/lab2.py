import re
from typing import List


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


def parse_abreviations(split_words: List[str]):
    # read abreviations
    with open("abreviations_names.txt", "r") as f:
        abreviations = f.read()

    # split abrevaitions in lines
    abreviations = abreviations.split("\n")

    i = 0
    while i < len(split_words) - 1:
        current_word = split_words[i]
        next_word = split_words[i+1]
        composed_word = current_word + next_word

        if composed_word.lower() in abreviations:
            whole_name = gather_compound_name(i+2, split_words)

            split_words[i] = composed_word + " " + whole_name
            split_words.pop(i+1)

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

        if (current_word == "-" and next_word == "ul") or not(len(previous_word) < 2 or len(next_word) < 3):
            # remove the hyphen
            split_words.pop(i)
            split_words.pop(i)
            # concatenate the next word with the current word
            split_words[i-1] += current_word + next_word

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


def tokenize(text: str):
    # read punctuation from "punctuation.txt"
    # with open("punctuation.txt", "r") as f:
    #     punctuation = f.read()

    # split
    words = re.findall(r"[\w']+|[\-.,!?;@:/()\]\[{}'\"]", text)

    print(words)
    words = parse_emails(words)
    words = parse_web_url(words)
    words = parse_ip(words)
    words = parse_abreviations(words)
    words = parse_compound_names(words)
    words = parse_hyphen(words)

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