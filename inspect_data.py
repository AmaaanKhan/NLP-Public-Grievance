import csv

path = r"citizen-grievances\citizen-grievances.csv"

with open(path, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)

    print("COLUMNS:")
    headers = next(reader)
    print(headers)

    print("\nFIRST 5 RECORDS:")
    for i, row in enumerate(reader):
        print(f"\nRecord {i + 1}:")
        for column, value in zip(headers, row):
            print(f"  {column}: {value[:250]}")

        if i == 4:
            break