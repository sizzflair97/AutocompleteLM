def contains_english(text):
    for char in text:
        # Check if the ASCII value of the character falls within the English alphabet range
        if 65 <= ord(char) <= 90 or 97 <= ord(char) <= 122:
            return True
    return False

def contains_only_korean(text):
    for char in text:
        # Check if the Unicode code point of the character is within the Hangul range
        if '\uAC00' <= char <= '\uD7A3':
            continue
        else:
            return False
    return True