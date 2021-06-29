from networkx.algorithms.bipartite.basic import color
from networkx.drawing.nx_pylab import draw
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt 

SEED8 = [0, 7, 3, 4, 1, 6, 2, 5]

#NOTE: All tournaments must have at least 2 players

def create_random_G(n):
    """
    Parameters:
        n - number of players in the tournament.

    This function creates a uniformly random tournament graph of a given size. The format is an n by n numpy array where the diagonal elements are 0 and the element on row i and column j is 1 if i beats j in the tournament and 0 otherwise. So, for every i and j, exactly one of the elements at (i,j) and (j,i) is 1. The random distribution is uniform over all graphs.
    """
    G = np.zeros((n,n))
    for a in range(n):
        for b in range(a):
            i, j = random.choice([(a,b), (b,a)]) #selects random winner
            G[i][j] = 1
    return G

def get_stationary_distribution(Q):
    """
    Parameters:
        Q - A transition probability matrix for a graph. This can be a list of lists or a numpy array.

    This function takes in a Markov transition probability matrix and computes the stationary probability distribution on the states. The output is a numpy list of the stationary probabilities at each state (the ith element in the output corresponds to state i). In the input, row i, column j corresonds to the probability that state j changes to state i. All this function is doing is finding a vector p such that Qp = p. It works by using numpy's eigenvalue/eigenvector function.

    The implementation is mostly taken from https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
    """
    evals, evecs = np.linalg.eig(Q)
    evec1 = evecs[:,np.isclose(evals, 1)]
    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so we want the real part.
    stationary = stationary.real
    return stationary

def get_co(G):
    """
    Parameters:
        G - a tournament graph matrix. This can be a list of lists or a numpy array.

    This function returns a list of Copeland scores for a given tournament. Element i in the output is the score for participant i.
    """
    return [int(sum(row)) for row in G]

def get_p(G):
    """
    Parameters:
        G - a tournament graph matrix. This can be a list of lists or a numpy array.

    This function returns a list of Markov scores for a given tournament (i.e. stationary distribution probabilities at each participant). Element i in the output is the score for participant i. This is the same as if the participants played many many "winner stays" matches where the loser is replaced by a random player and we counted what fraction of the wins each participant has.
    """
    n = len(G)
    diagCO = np.diag(get_co(G))
    Q = (G + diagCO)/(n-1)
    return get_stationary_distribution(Q)

def get_adjacency_list(G):
    """
    Parameters:
        G - a tournament graph matrix. This can be a list of lists or a numpy array.

    This function converts a tournament graph matric into a list of edges for the tournament. The edges are represented as tuples. An element (i, j) corresponds to an edge from i to j (i.e. i beats j).
    """
    n = len(G)
    lst = []
    for i in range(n):
        for j in range(n):
            if G[i][j] == 1:
                lst.append((i,j))
    return lst

def get_equipos(n):
    """
    Parameters:
        n - number of points
    
    This function takes in a number n returns a list of coordinates for n points equidistantly spaced out on the unit circle (centerd at 0). This is useful for drawing graphs with n vertices.
    """
    pos=[]
    for i in range(n):
        pos.append((np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)))
    return pos

def get_copeland_set_from_scores(co):
    """
    Parameters:
        co - a list of copeland scores

    This function takes in a list of Copeland scores and outputs a list of Copeland winners (the Copeland set). If i is in the list, then player i is a Copeland winner.
    """
    maxscore = max(co)
    return [i for i in range(len(co)) if co[i] == maxscore]

def get_markov_set_from_scores(p):
    """
    Parameters:
        p - a list of Markov scores

    This function takes in a list of Markov scores (probability distribution) and outputs a list of Markov winners (the Markov set). If i is in the list, then player i is a Markov winner.
    """
    maxscore = max(p)
    return [i for i in range(len(p)) if np.around(p[i]-maxscore, 4)==0]


def play_SE(G, seed = "default"):
    """
    Parameters:
        G - a tournament graph matrix. This can be a list of lists or a numpy array.
        seed - a seeding arrangement for the players. This must be either "default" or a list of the intigers in the range(0,n) in some order.

    This function takes in a tournament matrix and a seeding and plays a single elimination tournament. It returns a tuple (winner of the tournamnent, list of games played in the tournament) where each game is a tuple of two player numbers in some order. In each round, every other player plays the next player in the remaining list. The initial remaining list is the seeding. The default seeding is just the players in order. If there is an odd number of players in a round, then the last player in the remaining list get to move on to the next round without playing a game. Rounds are completed until there is one remaining player, which is declared the winner.
    """
    if seed == "default":
        seed = list(range(len(G)))
    
    remaining = seed
    played_games = []
    while len(remaining) > 1:
        new = []
        for i in range(0, len(remaining)-1, 2): #the -1 is to avoid crashes for odd n
            a = remaining[i+1]
            b = remaining[i]
            new.append([b,a][int(G[a][b])])
            played_games.append((a,b))
        if len(remaining)%2 == 1:
            new.append(remaining[-1])
        remaining = new
    return remaining[0], played_games


def draw_tourney(G,  copeland_set_color = None,  SE_winner_color = None, markov_set_color = None, labels = "default", SE_seed = "default", pos = "default"):
    """
    Parameters:
        G - a tournament graph matrix. This can be a list of lists or a numpy array.
        copeland_set_color - color for the Copeland set. If left as none, copeland set will not be calculated.
        SE_win_color - color for the single elimination bracket winner and played games highlight. If left as none, single elimination will not be played.
        markov_set_color - color for the Markov set. If left as none, Markov set will not be calculated
        labels - labels for the graph. Set to "default", "copeland", or "markov", or create custom label list.
        SE_seed - seeding for the SE tournament. Has no effect if SE_win_color is set to None. Set to "default", "random", or create a custom seeding.
        pos - the positions of the vertices in the tournament graph. Set to "default" or create a custom list of coordinate tuples.

    This function draws a tournamnet graph. The default seeding has the players in order. The random seeding is uniformly random. The default vertex positions is equidistantly placed around a unit circle. The default labels is the number of the players. Copeland labels are the Copeland scores. Markov labels are the Markov scores. Make sure that any custom seeding, positions, or labels have the right length (same as number of vertices) and format. The colored circles for the tournament solutions are of different sizes so that they can be seen when overlapping. 
    """

    #the following section calculates needed variables only if the settings require them
    n = len(G) #number of vertices
    if labels == "markov" or markov_set_color != None:
        p = get_p(G) # markov probabilities for each vertex
    if labels == "copeland" or copeland_set_color != None:
        co = get_co(G)  # copeland scores for each vertex
    
    #sets default positions
    if pos == "default":
        pos = get_equipos(n)
    
    # sets built-in label options
    if labels == "default":
        labels = {i: i for i in range(n)}
    elif labels == "copeland":
        labels = {i : co[i] for i in range(n)}
    elif labels == "markov":
        labels = {i : np.around(p[i], 3) for i in range(n)}
    
    nxG = nx.MultiDiGraph()
    nxG.add_edges_from(get_adjacency_list(G))
    plt.figure(figsize=(7,7)) # 7, 7 is the size of the output window
    nx.draw_networkx_edges(nxG, pos, width = 1, arrowsize = 10, arrows=True, min_source_margin=20, min_target_margin=20)
    nx.draw_networkx_nodes(nxG, pos, node_size=1000, node_color="white", edgecolors="black")

    if copeland_set_color != None:
        copeland_set = get_copeland_set_from_scores(co)
        nx.draw_networkx_nodes(nxG, pos, nodelist = copeland_set, node_size = 1000, node_color = copeland_set_color, edgecolors = "black")

    if markov_set_color != None:
        markov_set = get_markov_set_from_scores(p)
        nx.draw_networkx_nodes(nxG, pos, nodelist = markov_set, node_size=500, node_color = markov_set_color)

    if SE_winner_color != None:
        if SE_seed == "default":
            SE_seed =list(range(n))
        if SE_seed == "random":
            SE_seed =list(range(n))
            random.shuffle(SE_seed)

        SE_winner, SE_games = play_SE(G, SE_seed)
        nx.draw_networkx_edges(nxG, pos, edgelist=SE_games, width = 5, arrows=False, edge_color=SE_winner_color, alpha=0.3, min_source_margin=20, min_target_margin=20)
        nx.draw_networkx_nodes(nxG, pos, nodelist = [SE_winner], node_size=200, node_color=SE_winner_color)

    nx.draw_networkx_labels(nxG, pos, labels, font_size=10)
    plt.axis("off")
    plt.show()


#SOME EXAMPLES: 

# example 1 creates 10 random tournaments, each with 8 players, where the seeding is the standard fair matching if the player numbers were rankings
"""
for _ in range(10):
#     G = create_random_G(8)
#     draw_tourney(G, markov_set_color="red", labels="markov", copeland_set_color="yellow", SE_winner_color="blue", SE_seed = SEED8)
"""

# example 2 is a tournament with 5 players and random seeding
"""
G = [[0, 1, 1, 1, 0],[0, 0, 1, 1, 1],[0, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 0, 0, 0, 0]]
# draw_tourney(G, markov_set_color="red", labels="markov", copeland_set_color="yellow", SE_winner_color="blue", SE_seed="random")
"""
# 