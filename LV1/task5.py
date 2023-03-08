data = open("lv1/SMSSpamCollection.txt", encoding="utf8")

ham_count = 0
ham_words_count = 0
spam_count = 0
spam_words_count = 0
exclamation_count = 0

for line in data:
    words = line.split()
    if words[0]=="ham":
       ham_count +=1
       ham_words_count += len(words)-1
    if words[0]=="spam":
       spam_count +=1
       spam_words_count += len(words)-1
       last_word = words[-1]
       if last_word[-1]=="!":
          exclamation_count +=1

data.close()
print("Average number of words in ham messages:", int(ham_words_count/ham_count))
print("Average number of words in spam messages:", int(spam_words_count/spam_count))
print("Number of spam messages that end with exclamation:", exclamation_count)

