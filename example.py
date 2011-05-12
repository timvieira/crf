
import re
from nlp.evaluation import F1
from nlp.annotation import fromSGML, extract_contiguous
from iterextras import partition, iterview

from stringcrf import Instance, StringCRF, build_domain


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

        f1(train, name='TRAIN')
        f1(test, name='TEST')

        print
        print 'likelihood:', sum(map(crf.likelihood, iterview(train))) / len(train)
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

    fit = [crf.sgd, crf.perceptron][1]
    fit(train, iterations=iterations, validate=validate)

    if save:
        crf.save(save)


if __name__ == '__main__':

    def quicky(iterations=5):
        main(proportion=[0.2, 0.1], iterations=int(iterations), save=False)

    def quicky2(iterations=5):
        """ Like quicky except it will save the model. """
        main(proportion=[0.2, 0.1], iterations=int(iterations))

    def load(load='model.pkl~'):
        main(proportion=[0.7, 0.3], load=load)

    def run():
        main(proportion=[0.7, 0.3])

    def profile():
        from profiling.utils import profile_viz
        profile_viz('main(proportion=[0.1, 0.1], iterations=2, save=False)')

    from automain import automain;
    automain(available=[quicky, run, profile, load, quicky2],
             ultraTB=True)
