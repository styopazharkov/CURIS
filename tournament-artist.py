import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt 
import queue

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

def create_regular_G(n):
    """
    Parameters:
        n - number of players in the tournament.

    This function creates a regular tournament. I.e a tourney where everyone beats the next (n-1)/2 players. If n is even, then the first n/2 players beat n/2 players and the last n/2 players beat n/2 - 1 players.
    """
    G = np.zeros((n,n))
    for row in range(n):
        for col in range(n):
            # complicated line; the second and third part basically just makes sure only the first n/2 playes have outdegree n/2 when n is even
            if (col-row-1) % n < (n - 1)/2 and not (n % 2 == 0 and  row - col == n/2): 
                G[row][col]= 1
    return G

def create_strong0_G(n):
    """
    Parameters:
        n - number of players in the tournament. Must be a power of 2.

    This function creates a strong0 tournament. I.e a tourney where i beats j if i>j except 0 beats just enough strong players to win an SE bracket
    """
    G = np.zeros((n,n))
    for row in range(1, n):
        for col in range(row):
            G[row][col] = 1
    i = 1
    while i < n:
        flip_edge(G, i, 0)
        i = (i << 1) + 1
    return G


            

def flip_edge(G, i, j):
    """
    Parameters:
        G - a tournament graph matrix. This can be a list of lists or a numpy array.
        i, j - vertices of G (must be numbers in range(len(G)))

    This funtion modifies G so that the edge from i to j is flipped. The vertices i and j can be passed in in any order.
    """
    G[i][j], G[j][i] = G[j][i], G[i][j]

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

def play_PingPong(G, seed = "default", numgames = 100, passon = "random"):
    """
    Parameters:
        G - a tournament graph matrix. This can be a list of lists or a numpy array.
        seed - a seeding arrangement for the players. This must be either "default" or a list of the intigers in the range(0,n) in some order.
        numgames - number of games to play in the pingpong tournament
        passon - method of determining the next player to replace the loser. Set to "line" or to "random"

    This function takes in a tournament matrix and a seeding and plays a ping pong tournament. It returns a list of winners, a list of played games, and a list of scores of the players. In each the winner stays. The loser is replaced either by the next person in line (if passon is set to "line") or by a random opponent (if passon is set to "random"). The winners are the players who won the most games.
    """
    n = len(G)
    if seed == "default":
        seed = list(range(n))

    current_player = seed[0]
    scores = {i: 0 for i in range(n)}
    games = []

    if passon == "line":
        line = queue.Queue()
        for i in seed[1:]:
            line.put(i)
        for game in range(numgames):
            opponent = line.get()
            if G[current_player][opponent]:
                games.append((current_player, opponent))
                scores[current_player] += 1
                line.put(opponent)
            else:
                games.append((opponent, current_player))
                scores[opponent] += 1
                line.put(current_player)
                current_player = opponent
    elif passon == "random":
        for game in range(numgames):
            opponent = random.randint(0,n-1)
            if G[current_player][opponent]:
                games.append((current_player, opponent))
                scores[current_player] += 1
            else:
                games.append((opponent, current_player))
                scores[opponent] += 1
                current_player = opponent

    top_score = max(scores.values())
    winners = [i for i in range(n) if scores[i] == top_score]
    scorelist =  [scores[i] for i in range(n)]
    return winners, games, scorelist

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

def draw_tourney(G,  copeland_set_color = None,  SE_winner_color = None, markov_set_color = None, pingpong_winner_color = None, labels = "default", SE_seed = "default", pingpong_seed = "default", pingpong_numgames = "default", pingpong_passon = "default", pos = "default", node_size = 1000):
    """
    Parameters:
        G - a tournament graph matrix. This can be a list of lists or a numpy array.

        copeland_set_color - color for the Copeland set. If left as none, copeland set will not be calculated.
        
        SE_win_color - color for the single elimination bracket winner and played games highlight. If left as none, single elimination will not be played.

        markov_set_color - color for the Markov set. If left as none, Markov set will not be calculated

        pingpong_winners_color - color of the pingpong winners of the tournament

        labels - labels for the graph. Set to "default", "copeland", "markov", or "pingpong" or create custom label list. Set to None for no labels.

        SE_seed - seeding for the SE tournament. Has no effect if SE_win_color is set to None. Set to "default", "random", or create a custom seeding.

        pingpong_seed - seeding for the ping pong tournament. Has no effect if pingpong_win_color is set to None. Set to "default", "random", or create a custom seeding.

        pingpong_numgames - number of games for pingpong tournament. The default is 2 times the number of players

        pos - the positions of the vertices in the tournament graph. Set to "default" or create a custom list of coordinate tuples.

        node_size - the size of the nodes. It's set to 1000 by default.

    This function draws a tournamnet graph. The default seeding has the players in order. The random seeding is uniformly random. The default vertex positions is equidistantly placed around a unit circle. The default labels is the number of the players. Copeland labels are the Copeland scores. Markov labels are the Markov scores. Pingpong labels are the pingpong scores. Make sure that any custom seeding, positions, or labels have the right length (same as number of vertices) and format. The colored circles for the tournament solutions are of different sizes so that they can be seen when overlapping. 
    """

    #the following section calculates needed variables only if the settings require them
    n = len(G) #number of vertices
    if labels == "markov" or markov_set_color != None:
        p = get_p(G) # markov probabilities for each vertex
    if labels == "copeland" or copeland_set_color != None:
        co = get_co(G)  # copeland scores for each vertex
    if SE_winner_color != None:
        if SE_seed == "default":
            SE_seed =list(range(n))
        if SE_seed == "random":
            SE_seed =list(range(n))
            random.shuffle(SE_seed)
        SE_winner, SE_games = play_SE(G, SE_seed)
    if labels == "pingpong" or pingpong_winner_color != None:
        if pingpong_seed == "default":
            pingpong_seed =list(range(n))
        if pingpong_seed == "random":
            pingpong_seed =list(range(n))
            random.shuffle(pingpong_seed)
        if pingpong_numgames == "default":
            pingpong_numgames = n*2
        if pingpong_passon == "default":
            pingpong_passon = "random"
        pingpong_winners, pingpong_games, pingpong_scores = play_PingPong(G, seed=pingpong_seed, numgames=pingpong_numgames, passon=pingpong_passon)

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
    elif labels == "pingpong":
        labels = {i : pingpong_scores[i] for i in range(n)}
    
    nxG = nx.MultiDiGraph()
    nxG.add_edges_from(get_adjacency_list(G))
    plt.figure(figsize=(7,7)) # 7, 7 is the size of the output window
    nx.draw_networkx_edges(nxG, pos, width = 1, arrowsize = 10, arrows=True, min_source_margin=10+node_size//100, min_target_margin=10+node_size//100)
    nx.draw_networkx_nodes(nxG, pos, node_size=node_size, node_color="white", edgecolors="black")

    if copeland_set_color != None:
        copeland_set = get_copeland_set_from_scores(co)
        nx.draw_networkx_nodes(nxG, pos, nodelist = copeland_set, node_size = node_size, node_color = copeland_set_color, edgecolors = "black")

    if markov_set_color != None:
        markov_set = get_markov_set_from_scores(p)
        nx.draw_networkx_nodes(nxG, pos, nodelist = markov_set, node_size=node_size//2, node_color = markov_set_color)

    if SE_winner_color != None:
        nx.draw_networkx_edges(nxG, pos, edgelist=SE_games, width = 5, arrows=False, edge_color=SE_winner_color, alpha=0.3, min_source_margin=20, min_target_margin=20)
        nx.draw_networkx_nodes(nxG, pos, nodelist = [SE_winner], node_size=node_size//5, node_color=SE_winner_color)

    if pingpong_winner_color != None:
        nx.draw_networkx_edges(nxG, pos, edgelist=pingpong_games, width = 5, arrows=False, edge_color=pingpong_winner_color, alpha=0.2, min_source_margin=20, min_target_margin=20)
        nx.draw_networkx_nodes(nxG, pos, nodelist = pingpong_winners, node_size=node_size//7, node_color=pingpong_winner_color)


    if labels != None:
        nx.draw_networkx_labels(nxG, pos, labels, font_size=10)
    plt.axis("off")
    plt.show()


#SOME EXAMPLES: 

# example 1 creates 10 random tournaments, each with 8 players, where the seeding is the standard fair matching if the player numbers were rankings
"""
for _ in range(20):
    G = create_random_G(8)
    draw_tourney(G, markov_set_color="red", labels="markov", copeland_set_color="yellow", SE_winner_color="blue")
"""


# example 2 is a tournament with 5 players and random seeding
"""
G = [[0, 1, 1, 1, 0],[0, 0, 1, 1, 1],[0, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 0, 0, 0, 0]]
# draw_tourney(G, markov_set_color="red", labels="markov", copeland_set_color="yellow", SE_winner_color="blue", SE_seed="random")
"""

# example 3 is a vanilla regular tournament with 13 players and one edge flipped
"""
G = create_regular_G(13)
flip_edge(G, 0, 1)
draw_tourney(G, labels= None, node_size= 200)
"""

# example 4 is a graph where the markov set does not intersect with the copeland set. 
"""
G = [
    [0] + 10*[1] + 5 * [0],
    [0]*2 + [1,1,1] + [0]*3 + [1]*8,
    [0]*3 + [1,1,1] + [0]*2 + [1]*8,
    [0]*4 + [1,1,1] + [0]*1 + [1]*8,
    [0]*5 + [1,1,1] + [0]*0 + [1]*8,
    [0]+[1]+ [0]*4 + [1,1] + [1]*8,
    [0]+[1,1]+ [0]*4 + [1] + [1]*8,
    [0]+[1,1,1]+ [0]*4 + [1]*8,

    [0]+[0]*7+[0]+[1]*4+[0]*3,
    [0]+[0]*7+[0]*2+[1]*4+[0]*2,
    [0]+[0]*7+[0]*3+[1]*4+[0]*1,
    [1]+[0]*7+[0]*4+[1]*4+[0]*0,
    [1]+[0]*7+[0]*5+[1]*3,
    [1]+[0]*7+[1]*1+[0]*5+[1]*2,
    [1]+[0]*7+[1]*2+[0]*5+[1]*1,
    [1]+[0]*7+[1]*3+[0]*5+[1]*0
]
draw_tourney(G,  copeland_set_color="yellow", markov_set_color="red", labels="markov")
"""

# TRASH FOR TESTING BELOW THIS LINE: 

# for _ in range(10):
#     G = create_random_G(8)
#     draw_tourney(G, pingpong_winner_color="green", labels= "pingpong", pingpong_seed="random", pingpong_passon="line", pingpong_numgames=800, SE_winner_color="blue", markov_set_color="red")
# flip_edge(G, 0, 1)
# draw_tourney(G, labels="copeland")

G = create_strong0_G(16)
draw_tourney(G, SE_winner_color= "blue", labels="markov", markov_set_color="red")
# p = get_p(G)
# print(max(p))
# print(p[0])


