import json


chunks_description = []
chunks_specs = []

with open('../1-raw-data/products_updated.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for product in data:

    # find paragraphs
    paragraphs = product['description'].split('\n\n')

    for paragraph in paragraphs:

        chunk = {
            'document': paragraph,
            'metadata': {
                'id': product['id'],
                'title': product['title'],
                'url': product['url']
            }
        }

        chunks_description.append(chunk)

    # find specs
    specs = product['specs']

    for spec in specs:
        chunk = {
            'document': f"{spec['key']}: {spec['value']}",
            'metadata': {
                'id': product['id'],
                'title': product['title'],
                'key': spec['key'],
                'value': spec['value']
            }
        }

        chunks_specs.append(chunk)


# wirte jsonl
with open('chunks_description.jsonl', 'w', encoding='utf-8') as f:
    for chunk in chunks_description:
        json_line = json.dumps(chunk, ensure_ascii=False)
        f.write(json_line + '\n')

with open('chunks_specs.jsonl', 'w', encoding='utf-8') as f:
    for chunk in chunks_specs:
        json_line = json.dumps(chunk, ensure_ascii=False)
        f.write(json_line + '\n')