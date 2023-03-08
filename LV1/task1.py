def total_euro(working_hours, hourly_pay):
    return working_hours*hourly_pay
    
print("Enter your working hours: ")
working_hours = int(input())
print("Enter your hourly pay: ")
hourly_pay = int(input())
print("Your paycheck is:", total_euro(working_hours, hourly_pay))



