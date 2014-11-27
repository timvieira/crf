import re
import numpy as np
from arsenal.nlp.evaluation import F1
from arsenal.nlp.annotation import fromSGML, extract_contiguous
from arsenal.iterextras import partition, iterview

#from stringcrf import Instance, StringCRF, build_domain
from stringcrf2 import Instance, StringCRF, build_domain


def main(proportion=None, iterations=20, save='model.pkl~', load=None):

    class Token(object):
        def __init__(self, form):
            self.form = form
            self.attributes = []
        def add(self, features):
            """ Add features to this Token. """
            self.attributes.extend(features)

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
            for t in xrange(1, len(s)):
                s[t].add(f + '@-1' for f in token_features(s[t-1]))
            # next token features
            for t in xrange(len(s) - 1):
                s[t].add(f + '@+1' for f in token_features(s[t+1]))
        return s

    def get_data(f):
        for x in fromSGML(f, linegrouper="<NEW.*?>", bioencoding=False):
            x, y = zip(*[(Token(w), y) for y, w in x])
            preprocessing(x)
            yield Instance(x, truth=y)


    [train, test] = partition(get_data('tagged_references.txt'), proportion)


    def validate(model, iteration=None):

        def f1(data, name):
            print
            print 'Phrase-based F1:', name
            f1 = F1()
            for i, x in enumerate(iterview(data)):
                predict = extract_contiguous(model(x))
                truth = extract_contiguous(x.truth)
                # (i,begin,end) uniquely identifies the span
                for (label, begins, ends) in truth:
                    f1.add_relevant(label, (i, begins, ends))
                for (label, begins, ends) in predict:
                    f1.add_retrieved(label, (i, begins, ends))
            print
            return f1.scores(verbose=True)

        def weight_sparsity(W, t=0.0001):
            a = (np.abs(W) > t).sum()
            b = W.size
            print '%.2f (%s/%s) sparsity' % (a*100.0/b, a, b)

#        f1(train, name='TRAIN')
#        if test:
#            f1(test, name='TEST')

#        print
#        weight_sparsity(model.W)
        llh = sum(map(crf.likelihood, iterview(train, msg='llh'))) / len(train)

        from arsenal.viz.util import lineplot
        with lineplot('llh') as d:
            d.append(llh)

        print
        print 'likelihood:', llh
        print
        print

    if load:
        crf = StringCRF.load(load)
        validate(crf)
        return

    # Create and train CRF
    (L, A) = build_domain(train)

    print len(L), 'labels'
    print len(A), 'features'

    crf = StringCRF(L, A)

    if 0:
        print 'testing gradient....'
        crf.preprocess(train)
        crf.W[:] = np.random.uniform(-1,1,size=crf.W.shape)
        crf.test_gradient(train[:10])
        print 'testing....done'

    fit = [crf.sgd, crf.perceptron, crf.very_sgd][2]
    fit(train, iterations=iterations, validate=validate)

    if save:
        crf.save(save)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('action', choices=('quicky', 'quicky2','run','load'))
    parser.add_argument('iterations', type=int)
    args = parser.parse_args()

    if args.action == 'quicky':
        main(proportion=[0.2, 0.0], iterations=args.iterations, save=False)

    elif args.action == 'quicky2':
        main(proportion=[0.2, 0.1], iterations=args.iterations)

    elif args.action == 'load':
        main(proportion=[0.7, 0.3], load='model.pkl~')

    elif args.action == 'run':
        main(proportion=[0.7, 0.3], iterations=args.interations)
