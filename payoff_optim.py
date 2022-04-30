###################################
# Made by Aryan Gondkar
# Michigan State University
# CSE260 Honors Fall 2022
###################################
# Player tries to maximize gain for
# itself.
# Solves for (A,D,R) probabilities
# for the player
###################################

# Current state: Player(0) Enemy(0)
#
# Some standard payoff matrices are included

# WARNING: - Using inf throws double_scalar warning and gives unexpected results. Use very large numbers instead
#          - The code has not been optimized and might not scale well
#          - The primary goal for this code was for generating data for the project. Do not expect rigor and features

from cProfile import label
import random
from copy import deepcopy
from math import inf
from numpy import array, row_stack, transpose, matmul, array_equal, arange
from numpy import round as npround
from numpy.random import randn, seed
from matplotlib import pyplot as plt
import seaborn as sns

# region Reusability features
# Standard payoff matrices in (PLAYER_PAYOFF,ENEMY_PAYOFF) form
high_num=1E15
STANDARD_PAYOFFS = (
    (array(((0, 0, 0), (0, 0, 0), (1, 1, 1))), array(((0, 0, 0), (0, 0, 0), (1, 1, 1)))),  # S00
    (array(((high_num, -1, high_num), (0, 0, 0), (1, 1, 1))), array(((-high_num, 0, -high_num), (0, 0, 1), (0, 0, 1)))),  # Sn0
    (array(((-high_num, 0, -high_num), (0, 0, 1), (0, 0, 1))), array(((high_num, -1, high_num), (0, 0, 0), (1, 1, 1)))),  # S0n
    (array(((-1, -1, high_num), (0, 0, 0), (-high_num, 1, 1))), array(((-1, 0, -high_num), (-1, 0, 1), (high_num, 0, 1))))  # Snm
)
# endregion

# Parameters
PLOT_TITLE = "State (n,m)"
PLAYER_PAYOFF = STANDARD_PAYOFFS[3][0]
ENEMY_PAYOFF = STANDARD_PAYOFFS[3][1]
MAKE_PLOT = True

MUTATION_PROBABILITY = lambda x: 0.05
MUTATION_MULTIPLIER = lambda x: random.uniform(-0.1, 0.1)  # Function determining the scaling of change

ROUNDS = 500  # Rounds of evolution (default 500)
PLAYERS_COUNT = 50  # Set of players (default 50)
SELECT_TOP = 10  # How many are selected to parent (default 10)

FORCE_UNIQUE_STRATEGY = False  # Discards offspring if identical to parent (a simple method to encourage diversity of solutions)
MINIMIZE_OPPONENT_GAIN = True  # Should player focus on only maximizing his gain, or try minimizing the opponent's gain

VERBOSITY = 1
SEED = 0
# Variables
sns.set()
fig,ax=plt.subplots(2,1)
basis = (row_stack((1, 0, 0)), row_stack((0, 1, 0)), row_stack((0, 0, 1)))
players = []
enemies = []
random.seed(SEED)
seed(SEED)


class Player():
    value = 0
    strategy: row_stack

    def __init__(self, is_enemy=False, strategy=None) -> None:
        if not type(strategy)==array:
            self.strategy = abs(randn(3, 1))
        else:
            self.strategy = strategy

        if not is_enemy:
            players.append(self)
        else:
            enemies.append(self)

    def __lt__(self, other):
        return self.value < other.value

    def __get__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return array_equal(self.strategy, other.strategy)

    def __getitem__(self, key):
        if key == 0:
            return transpose(self.strategy)[0]
        elif key == 1:
            return self.value
        else:
            return None

    def __repr__(self):
        return str(f"Strategy: {transpose(self.strategy)[0]} Value:{self.value}\n")


def inject_custom_strategy(strat,enemy):
    Player(strategy=row_stack(strat),is_enemy=enemy)


def normalize_strategy(strat):
    sqrd = strat * strat
    return deepcopy(sqrd) / sum(sqrd)[0]


def mutate(player):
    for i in range(3):
        if random.random() < MUTATION_PROBABILITY(0):
            player.strategy = abs(player.strategy + \
                                  basis[i] * MUTATION_MULTIPLIER(0))


def make_players(elites=0, is_enemy=False):
    if is_enemy:
        l = enemies
    else:
        l = players

    while len(l) < PLAYERS_COUNT:
        if elites:
            parent = random.choice(elites)
            plr = deepcopy(parent)
            plr.value = 0
            mutate(plr)

            if FORCE_UNIQUE_STRATEGY and parent == plr:
                continue
            l.append(plr)
        else:
            Player(is_enemy)


def plot_strategies(elites):
    fig.suptitle(PLOT_TITLE)

    elite_p=elites[0]
    elite_e=elites[1]
    x_axis=arange(len(elite_p))

    p_characteristics=[[],[],[]]
    e_characteristics=[[],[],[]]


    for p in elite_p:
        for i in range(3):
            p_characteristics[i].append(p[0][i])

    ax[0].set_title("Top Player strategies")
    ax[0].set_ylim([0,1])
    ax[0].bar(x_axis-0.25,p_characteristics[0],label="Attack",width=0.25,color="#ef86a4")
    ax[0].bar(x_axis,p_characteristics[1],label="Defend",width=0.25,color="#59c7f2")
    ax[0].bar(x_axis+0.25,p_characteristics[2],label="Reload",width=0.25,color="#c6d65e")    
    ax[0].set_xticks(x_axis)
    ax[0].set_ylabel("Action Probability")
    ax[0].set_xlabel("Success Rank")

    fig.legend(loc="upper right")

    for e in elite_e:
        for i in range(3):
            e_characteristics[i].append(e[0][i])
    
    ax[1].set_title("Top Enemy strategies")
    ax[1].set_ylim([0,1])
    ax[1].bar(x_axis-0.25,e_characteristics[0],label="Attack",width=0.25,color="#ef86a4")
    ax[1].bar(x_axis,e_characteristics[1],label="Defend",width=0.25,color="#59c7f2")
    ax[1].bar(x_axis+0.25,e_characteristics[2],label="Reload",width=0.25,color="#c6d65e")
    ax[1].set_xticks(x_axis)
    ax[1].set_ylabel("Action Probability")
    ax[1].set_xlabel("Success Rank")

    fig.subplots_adjust(top=0.9)
    plt.tight_layout()


def run():  # Packaged for reusability
    global players
    global enemies
    players = enemies = []
    make_players()
    make_players(is_enemy=True)
    for rnd in range(ROUNDS):
        if VERBOSITY > 1:
            print("Round", rnd)

        for p_i in range(PLAYERS_COUNT):  # Match up each player with each enemy
            for e_i in range(PLAYERS_COUNT):
                p, e = players[p_i], enemies[e_i]
                p_score = matmul(matmul(transpose(p.strategy), PLAYER_PAYOFF), e.strategy)[0][0]
                e_score = matmul(matmul(transpose(e.strategy), ENEMY_PAYOFF), p.strategy)[0][0]

                if MINIMIZE_OPPONENT_GAIN:
                    p.value += p_score - e_score
                    e.value -= p_score - e_score  # The player's gain, is the enemy's loss
                else:
                    p.value += p_score
                    e.value += e_score

        if VERBOSITY > 1:
            print("\n")

        # Select make new generation with elite parents
        players.sort(reverse=True)
        players = players[:SELECT_TOP]
        elite_p = tuple(players)
        make_players(elite_p)

        enemies.sort(reverse=True)
        enemies = enemies[:SELECT_TOP]
        elite_e = tuple(enemies)
        make_players(elite_e, is_enemy=True)
        ##

    for p in elite_p:  # Normalize strategy vector for displaying
        p.strategy = normalize_strategy(p.strategy)
    for e in elite_e:
        e.strategy = normalize_strategy(e.strategy)

    if VERBOSITY:
        print("Strategy: (Attk, Dfnd, Reld) %")

        print("\nShowing player strategy-->")
        for p in elite_p[:5]:
            print(f"Strategy: {tuple(npround(p[0] * 100, 2))} %")

        print("\nShowing enemy strategy-->")
        for p in elite_e[:5]:
            print(f"Strategy: {tuple(npround(p[0] * 100, 2))} %")
    return (elite_p, elite_e)


if __name__ == "__main__":  # Perform project actions if not acting as module
    # inject_custom_strategy((1,0,0),False)
    # inject_custom_strategy((1,0,0),True)
    elites=run()
    if MAKE_PLOT:
        plot_strategies(elites)
        plt.show()
