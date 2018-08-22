"""Example usage of CRF module for citation segmentation. Note: This module is
just a quick demo of how to use the CRF module, not a serious attempt at a
state-of-the-art citation segmentation system. The data comes from Andrew
McCallum's webpage as you can see the README.

The goal of citation segmentation is very simple: Take a citation string, e.g.,

  A. Cau, R. Kuiper, and W.-P. de Roever. Formalising Dijkstra's development
  strategy within Stark's formalism. In C. B. Jones, R. C. Shaw, and T. Denvir,
  editors, Proc. 5th. BCS-FACS Refinement Workshop, 1992.

and segment it into labeled substrings corresponding to a handful of
bibliographic fields (author, title, year, editor, pages, etc.), e.g.,

  <author> A. Cau, R. Kuiper, and W.-P. de Roever. </author> <title> Formalising
  Dijkstra's development strategy within Stark's formalism. </title> <editor> In
  C. B. Jones, R. C. Shaw, and T. Denvir, editors, </editor> <booktitle>
  Proc. 5th. BCS-FACS Refinement Workshop, </booktitle> <date> 1992. </date>

We model this with a linear chain CRF by labeling each token as author, title,
year, editor, etc.

"""
import re
import numpy as np
from arsenal.nlp.evaluation import F1
from arsenal.nlp.annotation import fromSGML, extract_contiguous
from arsenal.iterextras import partition, iterview
from crf.stringcrf import Instance, Token, StringCRF, build_domain


def f1(data, name, model):
    f = F1()
    for i, x in enumerate(iterview(data, msg=f'F1 ({name})')):
        predict = extract_contiguous(model(x))
        truth = extract_contiguous(x.truth)
        # (i,begin,end) uniquely identifies the span
        for (label, begins, ends) in truth:
            f.add_relevant(label, (i, begins, ends))
        for (label, begins, ends) in predict:
            f.add_retrieved(label, (i, begins, ends))
    print()
    print('Phrase-based F1:', name)
    print()
    return f.scores(verbose=True)


def token_features(tk):
    """ very basic feature extraction. """
    w = tk.form
    yield 'word=' + w
    yield 'simplified=' + re.sub('[0-9]', '0', re.sub('[^a-zA-Z0-9()\.\,]', '', w.lower()))
    for c in re.findall('[^a-zA-Z0-9]', w):  # non-alpha-numeric
        yield 'contains(%r)' % c


def preprocessing(s):
    """ Run instance thru feature extraction. """
    s[0].add(['first-token'])
    s[-1].add(['last-token'])
    for tk in s:
        tk.add(token_features(tk))
    if 1:
        # previous token features
        for t in range(1, len(s)):
            s[t].add(f + '@-1' for f in token_features(s[t-1]))
        # next token features
        for t in range(len(s) - 1):
            s[t].add(f + '@+1' for f in token_features(s[t+1]))
    return s


def get_data(f):
    for x in fromSGML(f, linegrouper="<NEW.*?>", bioencoding=False):
        x, y = list(zip(*[(Token(w), y) for y, w in x]))
        preprocessing(x)
        yield Instance(x, truth=y)


def run(proportion=None, iterations=20):

    [train, _] = partition(get_data('data/tagged_references.txt'), proportion)

    def validate(model, _):
        llh = sum(map(crf.likelihood, iterview(train, msg='llh'))) / len(train)
        _,_,_,_,f = zip(*f1(train, 'train', model))
        overall = 100 * np.mean(f)   # equally weighted average F1
        print()
        print(f'log-likelihood: {llh:g}')
        print(f'F1 overall: {overall:.2f}')
        print()

    # Create and train CRF
    (L, A) = build_domain(train)
    crf = StringCRF(L, A)
    print(len(L), 'labels')
    print(len(A), 'features')

    fit = [crf.sgd, crf.perceptron, crf.very_sgd][0]
    fit(train, iterations=iterations, validate=validate)


def test_gradient(crf, data):
    from arsenal.maths.checkgrad import quick_fdcheck

    def grad():
        g = np.zeros_like(crf.W)
        for x in data:
            for k, v in crf.expectation(x).items():
                g[k] -= 1*v
            for k in x.target_features:
                g[k] += 1
        return g

    def func():
        return sum(crf.likelihood(x) for x in data)

    # Test random directions
    # Tim Vieira (2017) "How to test gradient implementations"
    # https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
    quick_fdcheck(func, crf.W, grad()) #.show()


def run_test():
    [train, _] = partition(get_data('data/tagged_references.txt'), [0.01, 0.0])
    (L, A) = build_domain(train)
    crf = StringCRF(L, A)

    print('Testing gradient of log-likelihood....')
    crf.preprocess(train)
    crf.W[:] = np.random.uniform(-1,1,size=crf.W.shape)
    test_gradient(crf, train)

    # Chekc that we have enough features to overfit this small training set.
    crf.sgd(train, iterations=10)

    llh = sum(map(crf.likelihood, iterview(train, msg='llh'))) / len(train)
    print(f'log-likelihood {llh:g}')

    _,_,_,_,f = zip(*f1(train, 'train', crf))
    overall = 100 * np.mean(f)   # equally weighted average F1
    print(f'Overall F1 (train): {overall:.2f}')


def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('action', choices=('test', 'run'))
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--plot', action='store_true')

    args = p.parse_args()
    if args.action == 'test':
        run_test()
    else:
        run(proportion=[0.7, 0.3], iterations=args.iterations)


if __name__ == '__main__':
    main()
