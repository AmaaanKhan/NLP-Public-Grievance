import csv
from collections import Counter

path = r"citizen-grievances\citizen-grievances.csv"

rows = 0
categories = Counter()
subcategories = Counter()
statuses = Counter()
empty_subcategories = 0

with open(path, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows += 1

        category = row.get("Category", "").strip()
        subcategory = row.get("Sub Category", "").strip()
        status = row.get("Grievance Status", "").strip()

        if category:
            categories[category] += 1
        if subcategory:
            subcategories[subcategory] += 1
        else:
            empty_subcategories += 1
        if status:
            statuses[status] += 1

print("Total records:", rows)
print("Unique categories:", len(categories))
print("Unique subcategories:", len(subcategories))
print("Empty subcategories:", empty_subcategories)

print("\nTop categories:")
for name, count in categories.most_common(15):
    print(f"{count:>8}  {name}")

print("\nTop subcategories:")
for name, count in subcategories.most_common(15):
    print(f"{count:>8}  {name}")

print("\nStatuses:")
for name, count in statuses.most_common():
    print(f"{count:>8}  {name}")