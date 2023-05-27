import pickle as pk


def get_result(source, target):
    sentences = f'data/aste/{target}/test.txt'
    in_domain = f'log/results-aste-{target}.pickle'
    cross_domain = f'log/results-aste-{source}_2_{target}.pickle'
    lines = []
    with open(sentences) as f:
        for line in f:
            lines.append(line.strip())
    return pk.load(open(in_domain, 'rb')), pk.load(open(cross_domain, 'rb')), lines


def analyse(labels, in_domains, cross_domains, sentences):
    assert len(labels) == len(in_domains) == len(cross_domains)
    for i in range(len(labels)):
        in_d = set(in_domains[i])
        cross_d = set(cross_domains[i])
        if len(in_d) != len(cross_d) or len(in_d) != len(in_d & cross_d):
            print(i, sentences[i])
            print('in domain: ', in_d)
            print('cross domain:', cross_d)
            print('label', labels[i])
            print()


def main():
    source = 'res14'
    target = 'lap14'
    in_domain, cross_domain, sentences = get_result(source, target)
    print(in_domain.keys())
    label = in_domain['labels']
    pred_cross = cross_domain['preds_fixed']
    pred_in = in_domain['preds_fixed']
    analyse(label, pred_in, pred_cross, sentences)


if __name__ == '__main__':
    main()