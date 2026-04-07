# Python file basics.py created

#--- 1. DATA CONTAINERS ---

marks = [85, 92, 78, 95, 88]     #list(Mutable,Ordered,Data Stored in Vector Form)

coordinates = (10, 20)           #tuple(Immutable,Ordered)

unique_values = {1, 2, 3, 3, 4}  #set(Unordered, No Duplicates)

student = {
    "name": "Kush",
    "age": 21,
    "marks": 90
}                                #Dictionary(Key:Value format,Mutable)

age = 24                         #integer
price = 99.99                    #float
is_valid = True                  #Boolean

#--- 2. Processing Loop (Iteration & Validation) ---

for i in marks:
    print(f"Validating Data Point: {i}") #Disply the Elements of List

# --- 3. ANALYTICAL FUNCTIONS ---
def analyze_numbers(numbers):
    """Calculates key statistics used in Data Science."""
    min_val = min(numbers)
    max_val = max(numbers)
    avg_val = sum(numbers) / len(numbers)
    return min_val, max_val, avg_val

# Running the Analysis
results = analyze_numbers(marks)
print(f"\n--- Statistics Report ---")

"""This Shows the Statistics report od the marks 
list which helps to derive insights like High, Low & Average"""

print(f"Low: {results[0]} | High: {results[1]} | Average: {results[2]}")


