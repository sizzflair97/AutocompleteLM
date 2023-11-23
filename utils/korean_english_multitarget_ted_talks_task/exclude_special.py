import re

def exclude_special_characters(input_string):
    # Define the set of characters to exclude
    special_characters = "~@#$%^&*()-+=_\\|:;'\"<>/[]."
    
    # Create a regular expression pattern to match any of the special characters
    pattern = f"[{re.escape(special_characters)}]"
    
    # Use re.sub to replace matches with an empty string
    result_string = re.sub(pattern, '', input_string)

    result = False

    if input_string == result_string:
        result = True

    return result_string, result