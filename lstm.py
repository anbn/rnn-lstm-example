#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys


class LSTM(object):
    @staticmethod
    def sigmoid(x): 
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def onehot(n, i):
        x = np.zeros((n,1))
        x[i] = 1
        return x

    def __init__(self, num):
        self.n_x, self.n_h, self.n_y = num

        self.Wa = np.random.randn(self.n_h, self.n_x+self.n_h)*0.01
        self.Wi = np.random.randn(self.n_h, self.n_x+self.n_h)*0.01
        self.Wf = np.random.randn(self.n_h, self.n_x+self.n_h)*0.01
        self.Wo = np.random.randn(self.n_h, self.n_x+self.n_h)*0.01
        self.ba = np.random.randn(self.n_h, 1)
        self.bi = np.random.randn(self.n_h, 1)
        self.bf = np.random.randn(self.n_h, 1)
        self.bo = np.random.randn(self.n_h, 1)

        self.Wy = np.random.randn(self.n_y, self.n_h)*0.01
        self.by = np.zeros((self.n_y,1))
        

    def get_loss(self, inputs, targets, hprev=None, cprev=None):
        if hprev is None: hprev = np.zeros((self.n_h,1))
        if cprev is None: cprev = np.zeros((self.n_h,1))
        xs, cs, hs, ys, ps = {}, {}, {}, {}, {}
        a, i, f, o = {}, {}, {}, {} 
        hs[-1] = np.copy(hprev)
        cs[-1] = np.copy(cprev)
        loss = 0

        for t in xrange(len(inputs)):
            xs[t] = self.onehot(self.n_x, inputs[t])

            xh = np.vstack((xs[t], hs[t-1]))
            a[t] = np.tanh(np.dot(self.Wa, xh) + self.ba)
            i[t] = self.sigmoid(np.dot(self.Wi, xh) + self.bi)
            f[t] = self.sigmoid(np.dot(self.Wf, xh) + self.bf)
            o[t] = self.sigmoid(np.dot(self.Wo, xh) + self.bo)
            cs[t] = i[t] * a[t] + f[t]*cs[t-1]
            hs[t] = o[t] * np.tanh(cs[t])

            ys[t] = np.dot(self.Wy, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t],0])
            
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dWa, dWi, dWf, dWo = np.zeros_like(self.Wa), np.zeros_like(self.Wi), \
                             np.zeros_like(self.Wf), np.zeros_like(self.Wo)
        dba, dbi, dbf, dbo = np.zeros_like(self.ba), np.zeros_like(self.bi), \
                             np.zeros_like(self.bf), np.zeros_like(self.bo)
        dc_next = np.zeros((self.n_h,1))
        dh_next = np.zeros((self.n_h,1))

        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWy += np.dot(dy, hs[t].T)
            dby += dy

            dh = np.dot(self.Wy.T, dy) + dh_next
            do = dh * np.tanh(cs[t])
            dc = dh * o[t] * (1.-np.tanh(cs[t])**2) + dc_next
            da = dc * i[t]
            di = dc * a[t]
            df = dc * cs[t-1]
            dc_next = dc * f[t]

            dha = da * (1. - a[t]*a[t])
            dhi = di * i[t] * (1. - i[t])
            dhf = df * f[t] * (1. - f[t])
            dho = do * o[t] * (1. - o[t])

            xh = np.vstack((xs[t], hs[t-1]))
            dWa += np.dot(dha, xh.T)
            dWi += np.dot(dhi, xh.T)
            dWf += np.dot(dhf, xh.T)
            dWo += np.dot(dho, xh.T)
            dba += dha
            dbi += dhi
            dbf += dhf
            dbo += dho

            acc = np.dot(self.Wa.T, dha) + \
                  np.dot(self.Wi.T, dhi) + \
                  np.dot(self.Wf.T, dhf) + \
                  np.dot(self.Wo.T, dho)
            dh_next = acc[self.n_x:]

        for dparam in [dWy, dWa, dWi, dWf, dWo, dby, dba, dbi, dbf, dbo]:
            np.clip(dparam, -5, 5, out=dparam)
        return loss, dWy, dWa, dWi, dWf, dWo, dby, dba, dbi, dbf, dbo


    def sample(self, hprev, cprev, seed_ix, n):
        x = self.onehot(self.n_x, seed_ix)
        ixes = [seed_ix]
        for t in xrange(n):
            xh = np.vstack((x, hprev))
            a = np.tanh(np.dot(self.Wa, xh) + self.ba)
            i = self.sigmoid(np.dot(self.Wi, xh) + self.bi)
            f = self.sigmoid(np.dot(self.Wf, xh) + self.bf)
            o = self.sigmoid(np.dot(self.Wo, xh) + self.bo)
            cprev = i * a + f*cprev
            hprev = o * np.tanh(cprev)

            y = np.dot(self.Wy, hprev) + self.by
            p = np.exp(y) / np.sum(np.exp(y))

            ix = np.random.choice(range(self.n_x), p=p.ravel())
            ixes.append(ix)
            x = self.onehot(self.n_x, ix)
        return ixes


    def gradCheck(self, inputs, targets, hprev, cprev):
        from random import uniform
        num_checks, delta = 10, 1e-5
        _, dWy, dWa, dWi, dWf, dWo, dby, dba, dbi, dbf, dbo = \
                lstm.get_loss(inputs, targets, hprev)
        for param,dparam,name in zip(
                [self.Wy, self.Wa, self.Wi, self.Wf, self.Wo,
                 self.by, self.ba, self.bi, self.bf, self.bo],
                [dWy, dWa, dWi, dWf, dWo, dby, dba, dbi, dbf, dbo],
                ['Wy', 'Wa', 'Wi', 'Wf', 'Wo', 'by', 'ba', 'bi', 'bf', 'bo']):
            print name
            for i in xrange(num_checks):
              ri = int(uniform(0,param.size))
              old_val = param.flat[ri]
              param.flat[ri] = old_val + delta
              cg0, _,_,_,_,_,_,_,_,_,_ = self.get_loss(inputs, targets, hprev, cprev)
              param.flat[ri] = old_val - delta
              cg1, _,_,_,_,_,_,_,_,_,_ = self.get_loss(inputs, targets, hprev, cprev)
              param.flat[ri] = old_val
              grad_analytic = dparam.flat[ri]
              grad_numerical = (cg0 - cg1) / ( 2 * delta )
              rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
              print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)


if __name__ == "__main__":
    text = []
    with open("data/calendar.txt") as f:
        for line in f: text.append(line)

    sentences = [s.split() for s in text]
    words = [w for s in sentences for w in s]
    o2w = { i:w for i,w in enumerate(list(set(words))) }
    w2o = { w:i for i,w in enumerate(list(set(words))) }

    lstm = LSTM(num=(len(o2w), 128, len(o2w)))
    learning_rate = 1e-1
    p, steps = 0, 64

    if True: lstm.gradCheck([0,1,2,3,4],[1,2,3,4,5],None,None)

    mWy = np.zeros_like(lstm.Wy)
    mby = np.zeros_like(lstm.by)
    mWa, mWi, mWf, mWo = np.zeros_like(lstm.Wa), np.zeros_like(lstm.Wi), \
                         np.zeros_like(lstm.Wf), np.zeros_like(lstm.Wo)
    mba, mbi, mbf, mbo = np.zeros_like(lstm.ba), np.zeros_like(lstm.bi), \
                         np.zeros_like(lstm.bf), np.zeros_like(lstm.bo)

    for i in xrange(5000):
        if p==0 or p>len(words):
            hprev, cprev = None, None
            p = 0 

        so = [w2o[w] for w in words[p:p+steps]]
        inputs, targets = so[:-1], so[1:]

        loss, dWy, dWa, dWi, dWf, dWo, dby, dba, dbi, dbf, dbo = \
                lstm.get_loss(inputs, targets, hprev, cprev)
        if i%10==0: print "%3d  loss: %2.6f" %(i, loss)

        for param, mparam, dparam in zip(
                [lstm.Wy, lstm.Wa, lstm.Wi, lstm.Wf, lstm.Wo, \
                 lstm.by, lstm.ba, lstm.bi, lstm.bf, lstm.bo], 
                [mWy, mWa, mWi, mWf, mWo, mby, mba, mbi, mbf, mbo],
                [dWy, dWa, dWi, dWf, dWo, dby, dba, dbi, dbf, dbo]):
            mparam += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mparam + 1e-8)
        p = (p+steps)
    
    for i in ["Mo","Di","Mi","Do","Fr","Sa","So"]:
        hprev = np.zeros((lstm.n_h,1))
        cprev = np.zeros((lstm.n_h,1))
        sample_ix = lstm.sample(hprev, cprev, w2o[i], 100)
        for w in [o2w[o] for o in sample_ix]:
            if len(w)>2: print w
            else:        print w,
