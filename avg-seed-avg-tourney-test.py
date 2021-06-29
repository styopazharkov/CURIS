# NOTE: This file is a mess and the result isn't that useful. It approximates the approximation ratio of a single elimination tounament on a uniform distribution over all graphs by taking big samples. I'm really only keeping it here in case I need some of the logic I implemented later -styopa

import random
import math

# n = 2^m
def generate_random_tourney(m):
    G = [[random.choice([0,1]) for j in range(i)] for i in range(2**m)]
    return G

def getwinner(G, i, j):
    opts=[i,j]
    if i<j:
        return opts[G[j][i]]
    else:
        return opts[::-1][G[i][j]]

def runtourney(G):
    a = list(range(len(G)))
    while len(a) > 1:
        b = []
        for i in range(0,len(a),2):
            b.append(getwinner(G, a[i], a[i+1]))
        a = b
    return a[0]

def maxscore(G):
    mx = 0
    for i in range(len(G)):
        mx = max(mx, getscore(G, i))
    return mx

def getscore(G, i):
    n = len(G)
    count = 0
    for j in range(n):
        if i != j and getwinner(G, i, j) == i:
            count += 1
    return count

def runtest(m, ntrials):
    ratiosum = 0
    selectedscoresum = 0
    maxscoresum = 0
    for i in range(ntrials):

        G = generate_random_tourney(m)
        winner = runtourney(G)
        selectedscore = getscore(G, winner)
        mxscore = maxscore(G)
        ratio = selectedscore/mxscore

        ratiosum += ratio
        maxscoresum += mxscore
        selectedscoresum += selectedscore

    # average score of the selected winner:
    print(selectedscoresum/ntrials, "which should be about", m/2+(2**m-1)/2)
    # average maxmum copeland score in the tournament
    print(maxscoresum/ntrials)
    # average ratio; ratio of averages
    print(ratiosum/ntrials, "which may be about", selectedscoresum/maxscoresum, ". The accuracy here is about", ratiosum/ntrials/selectedscoresum*maxscoresum)

runtest(3, 1000)