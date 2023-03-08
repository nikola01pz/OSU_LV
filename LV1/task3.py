import numpy as np

numbers = []

while True:
    value = input("Enter number or write Done: ")
    try:
       value = int(value)
    except ValueError:
       if(value=="Done"):
           break
       print("Sorry, enter number or write Done: ")
       continue
    numbers.append(value)
    
print("Numbers count:", len(numbers))
print("Arithmetic middle:", np.mean(numbers))
print("Min value is:", min(numbers))
print("Max value is:", max(numbers))
print("Sorted list:", np.sort(numbers))

