---
title: "A Regex Tutorial"
date: 2020-06-11
layout: single
author_profile: true
categories:
  - Regular Expression
tags: 
  - String Processing
excerpt: "Learn to process string operations in an efficient way"
mathjax: "true"
---
## Introduction
Recently I've been working on software developement projects and learning some NLP algorithms. Then regex came to my attention as a powerful string processing tool. It is so useful that I have to utilize the techniques almost everyday in my learning and work. Nonetheless, I think I might have a chance of leaving it to rot after some time. To maintain good memory of the syntax, I decided to create this blog, to both teach my future self and all of you interested readers. Without further a do, let's begin.

## Regex
- Full name: regular expression
- Create/search for a specific pattern in a string
- Very useful for text editing/file searching/phrase grouping/etc
- terminalogies
    - `raw string`: a raw string in python is just string prefixed with 'r', it tells python not to handle back slashes in any special way
    - `MetaCharacters`: symbols in the search pattern to create variated pattern
- Useful website: [1] [Regex101](https://regex101.com)<br>
- In the following, I'll outline all keywords used in the expression

### 1. Special sequences with backslash
- `\d`: digit(0-9)
- `\D`: Not a Digit(0-9)
- `\w`: Word character (a-z, A-Z, 0-9 _)
- `\W`: Not a word character
- `\s`: Whitespace(space, tab, blank line)


- `\b`: Word boundary
- `\B`: Not a word boundary
- `\A`: Start of string, __alternative representation__: `^`
- `\Z`: End of string, __alternative representation__: `$`
- `\g<id>`: Matches a previously defined group

### 2. MetaCharacters:
- `\`: Escape special characters
- `.`: Matches any character
- `$`: Matches end of string
- `^`: Matches start of string
- `[]`: Matches characters in brackets
- `[^ ]`: Matches characters not in brackets
- `|`: Either or
- `()`: Group

### 3. Quantifiers:
- `*`: 0 or more
- `+`: 1 or more
- `?`: 0 or 1
- `{m}`: Exactly m times
- `{n,}`: Min n times
- `{m,n}`: From m to n times, as many as possible
- `{m,n}?`: From m to n times, as few as possible

## Demonstrate the idea of raw string

```python
print('\tTab')
print(r'\tTab')
# this gives the following output
>>> 	Tab
>>> \tTab
```

Let's try out some regex usage by initializing the following strings
```python
import re

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890

Ha HaHa

MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )

coreyms.com

321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234

somebody123@gamil.com
test@test.com
some@qq.edu

Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T

FirstLast
First.Last
First Last
First..Last
First...Last
First....Last
'''
urls = '''
https://www.google.com
http://baidu.com
'''
sentence = 'Start a sentence and then bring it to an end'

```
Next let's apply 2 functions <kbd>test</kbd> and <kbd>test2</kbd> that we use to apply regex on `text_to_search` and `sentence` respectively:
```python
def test(s):
    pattern = re.compile(s)
    matches = pattern.finditer(text_to_search)

    for match in matches:
        print(match) # output: span(x, y) -> the location [x,y] in the string, match -> matched string
    print("")

def test2(s):
    pattern = re.compile(s)
    matches = pattern.finditer(sentence)

    for match in matches:
        print(match) # output: span(x, y) -> the location [x,y] in the string, match -> matched string
    print("")
```

Then we can try the following code and play around with it
```python
# use of \
test(r'ms\.com')
test(r'\\')
print("===========================")

# use of \d
test(r"\d")
print("===========================")

# use of \w
test(r"\w")
print("===========================")

# use of ^
test2(r'^Start')
test2(r'Start^a') # no match
test2(r'^a') # no match
print("===========================")

# use of $
test2(r'end$') # no match
test2(r'$Start') # no match
test2(r'a$') # no match
print("===========================")

# use word boundary usage
test(r'\bHa')
test(r'\BHa')
test(r'\bHa\b')
print("===========================")

# use of .
# warning: . does not takes \n into account
test(r'.')
test(r'.*')
print("===========================")

# usage of group ()
test(r'\d\d\d.\d\d\d.\d\d\d\d')
test(r'(\d*).\d\d\d.\d\d\d\d')
print("===========================")

# use of |
test(r'First( |\.)Last')
print("===========================")

# limit the value in pattern []
test(r'([6-9]\d*).(\d*).(\d*)')
test(r'[a-zA-Z]')
test(r'[^a-zA-Z]') # take NOT
test(r'First[ \.]Last')
print("===========================")

# limit the size in pattern using {}
test(r'(\d{3}).\d{3}.\d{4}')
test(r'\w{4,8}')
test(r'\w{4,8}?')
print("===========================")

# use of ?
test(r'First\.?Last') # has 1 or 0 .
test(r'First\.?\sLast')
test(r'555-?')
print("===========================")

# use of +
test(r'First\.+Last')
print("===========================")

# use of *
test(r'First\.*Last')
print("===========================")

# Now let's try a combination of the above methods
test(r'\d{3}[-.]\d{3}[-.]\d{4}')
```

### Substitution
To substitute existing characters in a string, we need to specify a pattern directly, and we then use the `.sub`  method.

```python
pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
subbed_urls = pattern.sub(r'\2\3',urls)
print(subbed_urls)
```

The `pattern` can also be used to direcly find all the occurances, so that we don't need to define <kbd>test</kbd> and <kbd>test2</kbd> by ouselves.

```python
pattern = re.compile(r'(\w+)-(\w+)-(\w+)')
matches1 = pattern.findall(text_to_search)
for match in matches1:
    print(match)
    
pattern2 = re.compile(r'\d{3}-\d{3}-\d{4}')
matches2 = pattern2.findall(text_to_search)
for match in matches2:
    print(match)

### Check match at the begin of the string
pattern = re.compile(r'Start')
matches = pattern.match(sentence)
print(matches)

pattern = re.compile(r'sentence')
matches = pattern.match(sentence)
print(matches)

### Case insensitive mode
pattern = re.compile(r'start', re.I) # re.I is a short version of re.IGNORECASE
matches = pattern.match(sentence)
print(matches)

### Actual functionality demo
# Email search
test(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9_.+-]+\.[a-zA-Z0-9_.+-]+')
test(r'(\w+)@(\w+)\.(\w+)')

pattern = re.compile(r'(\w+)@(\w+)\.(\w+)')
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match.group(1), match.group(2), match.group(3))
```