while True:
    number = input('Pick a number in range 0.0-1.0: ')
    try:
       number = float(number)
    except ValueError:
       print("Enter valid number")
       continue
    if 0.0<=number<=1.0:
        break
    else:
        print("Number is not in range")   

if(number>=0.9):
    print("A")
if(number>=0.8):
    print("B")
if(number>=0.7):
    print("C")
if(number>=0.6):
    print("D")
else:
    print("F")
