import re

my_string="Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"

sentence_endings = r"[.?!]" # Write a pattern to match sentence endings: sentence_endings
print(re.split(sentence_endings, my_string)) # Split my_string on sentence endings and print the result
Output - ["Let's write RegEx", "  Won't that be fun", '  I sure think so', 
'  Can you find 4 sentences', '  Or perhaps, all 19 words', '']


capitalized_words = r"[A-Z]\w+" # Find all capitalized words in my_string and print the result
print(re.findall(capitalized_words, my_string))
Output- ['Let', 'RegEx', 'Won', 'Can', 'Or']


spaces = r"\s+"
print(re.split(spaces, my_string)) # Split my_string on spaces and print the result
Output -["Let's", 'write', 'RegEx!', "Won't", 'that', 'be', 'fun?', 'I', 'sure', 
'think', 'so.', 'Can', 'you', 'find', '4', 'sentences?', 'Or', 'perhaps,', 'all', '19', 'words?']


digits = r"\d+"
print(re.findall(digits, my_string)) # Find all digits in my_string and print the result
Output - ['4', '19']

# To search the presence or absence of some pattern in your string you can use the re.match or re.search methods. re.search will
#try to look for pattern in the entire string where as re.match only looks at the presence of that pattern at the start 
#of the string

match = re.search("coconuts", scene_one)
match = re.match("coconuts", scene_one)


# In regex you can define a group using the () brackets
# Explicit charated ranges can be provided in [] brackets
# OR is represented using | charater 

match_digits_and_words_pattern= ('(\d+|\w+)')
re.findall(match_digits_and_words_pattern, "He has 11 cats.")
Output-  ['He', 'has', '11', 'cats']

# [a-zA-Z] upper and lower case alphabets
# [0-9] numbers 0 to 9
# [a-zA-Z\-\.] upper and lower case alphabets and - and .
#(a-z) a , - , z
# (\s+|,) space or comma
