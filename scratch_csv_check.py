import csv

terms = [
    "Ludwigsfelde", "Wannsee", "Werder", "Erkner", "Fuerstenwalde", "Fürstenwalde",
    "Bernau", "Eberswalde", "Oranienburg", "Koenigs Wusterhausen", "Königs Wusterhausen",
    "Zossen", "Rangsdorf", "Strausberg", "Lichtenberg", "Mahlsdorf",
    "Potsdam", "Gesundbrunnen", "Grossbeeren", "Großbeeren", "Birkengrund",
    "Lübben", "Cottbus"
]

found = {}

with open("Haltestellen_VBB/UMBW.CSV", mode='r', encoding='latin-1') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        if len(row) < 7: continue
        name = row[0]
        for term in terms:
            if term.lower() in name.lower():
                if term not in found:
                    found[term] = []
                found[term].append(name.strip())

for term, names in sorted(found.items()):
    print(f"\n=== '{term}' ===")
    for n in names[:10]:
        print(f"  {n}")
