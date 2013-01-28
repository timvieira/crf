from numpy import fromiter, int32

from arsenal.alphabet import Alphabet
from crf import CRF


class StringCRF(CRF):
    """
    Conditional Random Field (CRF) for linear-chain structured models with
    string-valued labels and features.
    """

    def __init__(self, label_alphabet, feature_alphabet):
        self.label_alphabet = label_alphabet
        self.feature_alphabet = feature_alphabet
        CRF.__init__(self, len(self.label_alphabet), len(self.feature_alphabet))

    def __call__(self, x):
        return self.label_alphabet.lookup_many(CRF.__call__(self, x))

    def preprocess(self, data):
        """
        preprocessing hook which caches the ``feature_table`` and ``target_features``
        attributes of a Instance.
        """
        A = self.feature_alphabet.map
        L = self.label_alphabet
        for x in data:
            # cache feature_table
            if x.feature_table is None:
                x.feature_table = f = {}
                F = x.F
                for i, y in L.enum():
                    f[0,None,i] = fromiter(A(F(0,None,y)), dtype=int32)
                for t in xrange(1, x.N):
                    for j, y in L.enum():
                        for i, yp in L.enum():
                            f[t,i,j] = fromiter(A(F(t,yp,y)), dtype=int32)
            # cache target_features
            if x.target_features is None:
                x.target_features = self.path_features(x, list(L.map(x.truth)))


def build_domain(data):
    """
    Do feature extraction to determine the set of *supported* featues, i.e.
    those active in the ground truth configuration and active labels. This
    function will each features and label an integer.
    """
    L = Alphabet()
    A = Alphabet()
    for x in data:
        L.add_many(x.truth)  # add labels to label domain
        # extract features of the target path
        F = x.F
        path = x.truth
        A.add_many(F(0, None, path[0]))
        A.add_many(k for t in xrange(1, x.N) for k in F(t, path[t-1], path[t]))
    # domains are now ready
    L.freeze()
    A.stop_growth()
    return (L, A)


class Instance(object):

    def __init__(self, s, truth=None):
        self.sequence = list(s)
        self.truth = truth
        self.N = len(s)
        # CRF will cache data here
        self.feature_table = None
        self.target_features = None

    def F(self, t, yp, y):
        """ Create the vector of active indicies for this label setting. """
        # label-label-token; bigram features
        #for f in self.sequence[t].attributes:
        #    yield '[%s,%s,%s]' % (yp,y,f)
        # label-token; emission features
        for f in self.sequence[t].attributes:
            yield '[%s,%s]' % (y,f)
        # label "prior" feature
        yield '[%s]' % y
        # label-label; transition feature
        yield '[%s,%s]' % (yp,y)
