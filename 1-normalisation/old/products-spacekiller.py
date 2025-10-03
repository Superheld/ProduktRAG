import json

with open('products_updated.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for product in data:
    if 'url' in product:
        product['url'] = product['url'].replace(' ', '')
    
    for spec in product['specs']:
        spec['key'] = spec['key'].replace('  /  ', ' / ')
        spec['value'] = spec['value'].replace('  /  ', ' / ')


with open('products_updated.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)