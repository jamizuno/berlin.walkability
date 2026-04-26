import csv

path = r'c:\Users\jermo\Documents\BERLINMAP-DISTANCE\Haltestellen_VBB\UMBW.CSV'
try:
    with open(path, mode='r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=';')
        for i, row in enumerate(reader):
            if i > 10: break
            print(row)
except Exception as e:
    print(e)
