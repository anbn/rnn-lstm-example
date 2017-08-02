#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class RNN(object):
    def __init__(self, num):
        self.n_x, self.n_h, self.n_y = num

        self.Wxh = np.random.randn(self.n_h, self.n_x)*0.01
        self.Whh = np.random.randn(self.n_h, self.n_h)*0.01
        self.Why = np.random.randn(self.n_y, self.n_h)*0.01
        self.bh = np.zeros((self.n_h,1))
        self.by = np.zeros((self.n_y,1))
        

    def get_loss(self, inputs, targets, hprev=None):
        if hprev is None:
            hprev = np.zeros((self.n_h,1))
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0

        for t in xrange(len(inputs)):
            xs[t] = np.zeros((self.n_x,1)) # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dhraw = (1 - hs[t] * hs[t]) * np.dot(self.Why.T, dy) + dhnext
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(self, hprev, seed_ix, n):
        x = np.zeros((self.n_x, 1))
        x[seed_ix] = 1
        ixes = [seed_ix]
        for t in xrange(n):
            hprev = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hprev) + self.bh)
            y = np.dot(self.Why, hprev) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.n_x), p=p.ravel())
            x = np.zeros((self.n_x, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes


if __name__ == "__main__":
    text = []
    text.append("red fox ate the bird .")
    text.append("blue fox chased the bird .")
    text.append("green fox fell asleep .")
    text.append("black fox was the bird .")
    text.append("purple fox built a wall .")
    text.append("yellow fox chased a fox .")

    sentences = [s.split() for s in text]
    words = list(set([w for s in sentences for w in s]))
    o2w = { i:w for i,w in enumerate(words) }
    w2o = { w:i for i,w in enumerate(words) }

    rnn = RNN(num=(len(o2w), 80, len(o2w)))

    smooth_loss = -np.log(1.0/len(words))*10 # loss at iteration 0
    learning_rate = 1e-1

    for i in xrange(1000):
        for s in text:
            hprev = np.zeros((rnn.n_h,1))
            so = [w2o[w] for w in s.split()]
            inputs, targets = so[:-1], so[1:]

            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = \
                    rnn.get_loss(inputs, targets, hprev)
            if i%100==0: print "%3d  loss: %2.6f" %(i, loss)

            for param, dparam in zip(
                    [rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by], 
                    [dWxh, dWhh, dWhy, dbh, dby]):
                param += -learning_rate * dparam
    
    for i in [sentence[0] for sentence in sentences]:
        hprev = np.zeros((rnn.n_h,1))
        sample_ix = rnn.sample(hprev, w2o[i], 10)
        print [o2w[o] for o in sample_ix]
