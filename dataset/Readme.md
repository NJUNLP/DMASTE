# Data Format
Sentence####[[a1, o1, s1], [a2, o2, s2],...]####Sub-Domain

* a: the start and end indexes of the aspect terms, if the aspect terms are implicit, it will be [-1]. If the aspect term is a single word, denoted by the subscript i, then it is represented as [i]. If the aspect term is a multi-word phrase, with the starting index denoted by i and the ending index denoted by j, then it is represented as [i, j].
* o: the start and end indexes of the opinion terms. If the opinion term is a single word, denoted by the subscript i, then it is represented as [i]. If the opinion term is a multi-word phrase, with the starting index denoted by i and the ending index denoted by j, then it is represented as [i, j].
* s: the sentiment polarity, including POS, NEG, NEU. 

File reading examples
```
    data = []
    with open(file_name) as f:
        for line in f:
            line = [x.split('####') for x in lines]
            sentence, triples,  = line[:2]
            triples = eval(triples)
            data.append([sentence, triples])
```
