import csv

path = r'c:\Users\jermo\Documents\BERLINMAP-DISTANCE\Haltestellen_VBB\UMBW.CSV'
search_term = "Hoppegarten"

try:
    with open(path, mode='r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=';')
        found = []
        for row in reader:
            if not row: continue
            name = row[0]
            if search_term.lower() in name.lower():
                found.append(row)
        
        for f in found:
            print(f)
except Exception as e:
    print(e)
