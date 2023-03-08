data = open("lv1/song.txt")
dictionary={}

for line in data:
    for word in line.split():
        if word not in dictionary:
            dictionary[word] = 1
    for word in line.split():
        if word in dictionary:
            dictionary[word] +=1

data.close()
print(dictionary)