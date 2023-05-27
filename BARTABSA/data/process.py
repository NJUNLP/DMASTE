import os 
import json
def convert_triples(triples, words):
    aspects = []
    opinions = []
    for i, triple in enumerate(triples):
        a, o, s = triple
        aspect = {'index': i, 'from': a[0], 'to': a[-1] + 1, 'polarity': s, 'term': words[a[0]: a[-1] + 1]}
        opinion = {'index': i, 'from': o[0], 'to': o[-1] + 1, 'term': words[o[0]: o[-1] + 1]}
        aspects.append(aspect)
        opinions.append(opinion)
    return aspects, opinions

def convert(input_file, output_file):
    dataset = []
    with open(input_file) as f:
        for line in f:
            ins = {}
            sent, triples = line.split('####')
            ins['raw_words'] = sent 
            ins['words'] = sent.split(' ')
            triples = eval(triples)
            ins['aspects'], ins['opinions'] = convert_triples(triples, ins['words'])
            dataset.append(ins)
    with open(output_file, 'w') as f:
        json.dump(dataset, f)


def main():
    root = '../../ia-dataset'
    for domain in os.listdir(root):
        domain_dir = f'{root}/{domain}'
        if '.' in domain:
            continue
        for mode_file in os.listdir(domain_dir):
            mode = mode_file.split('.')[0]
            file_name = f'{domain_dir}/{mode_file}'
            os.makedirs(f'{domain}', exist_ok=True)
            convert(file_name, f'./{domain}/{mode}.json')
        if 'train.json' not in os.listdir(domain):
            os.system('cp {}/test.json {}/train.json'.format(domain, domain))
        


main()