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

