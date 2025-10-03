#!/usr/bin/env python3
import json
import csv
import sys

def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("Keine Daten in der JSON-Datei gefunden.")
        return

    # Alle möglichen Spezifikationsschlüssel sammeln
    all_spec_keys = set()
    for item in data:
        if 'specs' in item and item['specs']:
            for spec in item['specs']:
                all_spec_keys.add(spec['key'])

    # CSV-Header erstellen
    headers = ['id', 'url', 'title', 'description']
    spec_headers = sorted(list(all_spec_keys))
    headers.extend(spec_headers)

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for item in data:
            row = [
                item.get('id', ''),
                item.get('url', ''),
                item.get('title', ''),
                item.get('description', '')
            ]

            # Spezifikationen als Dict für schnellen Zugriff
            specs_dict = {}
            if 'specs' in item and item['specs']:
                for spec in item['specs']:
                    specs_dict[spec['key']] = spec['value']

            # Spezifikationswerte hinzufügen
            for spec_key in spec_headers:
                row.append(specs_dict.get(spec_key, ''))

            writer.writerow(row)

    print(f"CSV-Datei erfolgreich erstellt: {csv_file}")
    print(f"Anzahl der Datensätze: {len(data)}")
    print(f"Anzahl der Spalten: {len(headers)}")

if __name__ == "__main__":
    json_file = "1-raw-data/products.json"
    csv_file = "products.csv"

    try:
        json_to_csv(json_file, csv_file)
    except FileNotFoundError:
        print(f"Datei {json_file} nicht gefunden.")
    except json.JSONDecodeError:
        print("Fehler beim Parsen der JSON-Datei.")
    except Exception as e:
        print(f"Fehler: {e}")