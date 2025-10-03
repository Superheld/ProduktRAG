#!/usr/bin/env python3
import csv
import json
import sys

def csv_to_json(csv_file, json_file):
    data = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Basis-Produktdaten
            product = {
                'id': row.get('id', ''),
                'url': row.get('url', ''),
                'title': row.get('title', ''),
                'description': row.get('description', ''),
                'specs': []
            }

            # Alle anderen Spalten als specs behandeln
            for key, value in row.items():
                if key not in ['id', 'url', 'title', 'description'] and value.strip():
                    product['specs'].append({
                        'key': key,
                        'value': value.strip()
                    })

            data.append(product)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"JSON-Datei erfolgreich erstellt: {json_file}")
    print(f"Anzahl der Datensätze: {len(data)}")

if __name__ == "__main__":
    csv_file = "1-raw-data/products.csv"
    json_file = "products_updated.json"

    try:
        csv_to_json(csv_file, json_file)
    except FileNotFoundError:
        print(f"Datei {csv_file} nicht gefunden.")
    except Exception as e:
        print(f"Fehler: {e}")