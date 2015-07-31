#import warnings
#import numpy as np
#from crf import logsumexp, defaultdict, exp, add, iterview
#
#class CRF2(object):
#
#    def risk(self, x, Y):
#        """
#        Risk (Hamming loss)
#        """
#
#        warnings.warn('this implementation is incorrect!')
#
#        Y = self.label_alphabet.map(Y)
#
#        N = x.N; K = self.K; f = x.feature_table
#        (g0, g) = self.log_potentials(x)
#
#        a = self.forward(g0,g,N,K)
#        b = self.backward(g,N,K)
#
#        # log-normalizing constant
#        logZ = logsumexp(a[N-1,:])
#
#        Er = 0.0
#        Erf = defaultdict(float)
#        Ef = defaultdict(float)
#
#        # The first factor needs to be special case'd
#        c = exp(g0 + b[0,:] - logZ)
#        for y in xrange(K):
#            p = c[y]
#            #if p < 1e-8: continue
#            r = (Y[0] != y)
#            Er += p * r
#            for k in f[0, None, y]:
#                Ef[k] += p
#                Erf[k] += p*r
#
#        for t in xrange(1,N):
#            # vectorized computation of the marginal for this transition factor
#            c = exp((add.outer(a[t-1,:], b[t,:]) + g[t-1,:,:] - logZ))
#
#            for yp in xrange(K):
#                for y in xrange(K):
#                    p = c[yp, y]
#                    #if p < 1e-8: continue
#                    r = (Y[t] != y)
#                    Er += p * r
#                    for k in f[t, yp, y]:
#                        Ef[k] += p
#                        Erf[k] += p*r
#
#        Cov_rf = defaultdict(float)
#        for k in Ef:
##            if abs(Ef[k] - Erf[k]) > 1e-8:
##                print k, Ef[k], Erf[k]
#            Cov_rf[k] = Erf[k] - Er*Ef[k]
#
#        return Er, Cov_rf
#
#    def test_gradient_risk(self, data, subsetsize=10):
#
#        def fd(x, i, eps=1e-4):
#            """Compute `i`th component of the finite-difference approximation to the
#            gradient of log-likelihood at current parameters on example `x`.
#
#            """
#
#            was = self.W[i]   # record value
#
#            self.W[i] = was+eps
#            b,_ = self.risk(x, x.truth)
#
#            self.W[i] = was-eps
#            a,_ = self.risk(x, x.truth)
#
#            self.W[i] = was   # restore original value
#
#            return (b - a) / 2 / eps
#
#        for x in iterview(data, msg='test grad'):
#
#            _, g = self.risk(x, x.truth)
#
#            # pick a subset of features to test
#            d = np.random.choice(g.keys(), subsetsize, replace=0)
#
#            f = {}
#            for i in iterview(d, msg='fd approx'):     # loop over active features
#                f[i] = fd(x, i)
#
#            from arsenal.math import compare
#            compare([f[k] for k in d],
#                    [g[k] for k in d], name='test gradient %s' % x,
#                    scatter=1,
#                    show_regression=1,
#                    alphabet=d)
#            import pylab as pl
#            pl.show()
