import sys

mydict = {}
keys = []

f = open(sys.argv[1], 'r')
#open('words.txt', 'r')

def add_words(word, words_dict):
    if word in words_dict:
        words_dict[word] += 1
    else:
        words_dict[word] = 1
        keys.append(word)

'''
def print_result(words_dict):
    i = 0
    for key, val in words_dict.items():
        print(key + ' ' + repr(i) + ' ' + repr(val) + '\n')
        i += 1
'''

for line in f:
    words_list = line.split(" ")
    for words in words_list:
        add_words(words, mydict)

#print_result(mydict)
f2 = open('Q1.txt', 'w+')

for i in keys:
    if keys.index(i) <= 298:
        f2.write("{} {} {}\n".format(i, repr(keys.index(i)), repr(mydict[i])))
    else:
        f2.write(i.strip() + ' ' +  repr(keys.index(i)) + ' ' + repr(mydict[i]))


#for key, val in mydict.items():


f2.close()

            
#print(words)

