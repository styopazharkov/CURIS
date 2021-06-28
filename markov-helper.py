from networkx.drawing.nx_pylab import draw
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt 


#creates random directed graph matrix. i, j is 1 if i beats j. 0 otherwise
def create_random_G(n):
    G = np.zeros((n,n))
    for a in range(n):
        for b in range(a):
            i, j = random.choice([(a,b), (b,a)]) #selects random winner
            G[i][j] = 1
    return G

#returns diagonal matrix of copeland scores
def find_diagCO(G):
    n = len(G)
    diagCO = [[0 for _ in range(n)] for _ in range(n)]
    COlist = [sum(row) for row in G]
    return np.diag(COlist)

# creates the markov move probability matrix from G and the diagonal copeland score matrix
def find_Q(G, diagCO):
    n = len(G)
    return (G + diagCO)/(n-1)

# finds stable probability distribution from move probability matrix. I.e. finds p st. pQ = p. 
# From https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
def find_p(Q):
    evals, evecs = np.linalg.eig(Q)
    evec1 = evecs[:,np.isclose(evals, 1)]
    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]


    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real
    return stationary

# converts from adjacency matrix to adjacency list
def get_adjacency_list(G):
    n = len(G)
    lst = []
    for i in range(n):
        for j in range(n):
            if G[i][j] == 1:
                lst.append((i,j))
    return lst

#gets coordinates of points around a circle
def get_pos(n, radius):
    pos=[]
    for i in range(n):
        pos.append((radius * np.cos(2*np.pi*i/n), radius * np.sin(2*np.pi*i/n)))
    return pos


def get_color_from_probability(probability):
    rounded_decimal_val = int((probability**0.3)*256)
    #TODO: fix this function
    hexval = hex(rounded_decimal_val)[2:]
    if len(hexval) == 1:
        hexval = "0"+hexval
    
    return "#8000"+hexval

#draws a tournament with the stationary probabilities as the labels
def draw_tourney(G, p):
    n = len(G)
    nxG = nx.MultiDiGraph()
    nxG.add_edges_from(get_adjacency_list(G))
    pos = get_pos(n, 5)
    color_map=[get_color_from_probability(p[i]) for i in range(n)]
    labels = {i : np.around(p[i], 3) for i in range(n)}

    plt.figure(figsize=(5,5))
    nx.draw(nxG, pos, labels = labels, with_labels=True, node_size=1000, node_color="white", width = 1, arrowsize = 10, font_size=10)
    plt.show()


G = create_random_G(9)
diagCO = find_diagCO(G)
Q = find_Q(G, diagCO)
print(G)
p = find_p(Q)
draw_tourney(G,p)

