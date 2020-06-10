import random
from math import sqrt, log
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Blackjack microchallenge:
# https://www.kaggle.com/learn/microchallenges

# Simplified blackjack with replacement.
# Draws (even at 21) go to dealer.
# One of dealer's cards is observable.
# Dealer hits until 17.

# Distribution of cards in
# https://github.com/Kaggle/learntools/blob/master/learntools/python/blackjack.py
def deal():
    return random.choice(list(range(2, 11)) + [10, 10, 10, 10])


# Each bandit corresponds to a state:
# state = (player total, number of player aces, dealer's card).
# Arm 0 is stay, Arm 1 is hit.

states = [(t, a, c) for t in range(4, 31+1) for a in range(21+1) for c in range(2, 11+1)]

def result(state):
    player = state[0]
    aces = state[1]
    while player > 21 and aces > 0:
        player -= 10
        aces -= 1
    if player > 21:
        return 0
    dealer = state[-1]
    while dealer < 17:
        dealer += deal()
    return int( dealer > 21 or player > dealer )

print("num bandits: {}".format(len(states)))

bandits = dict()

def ucb1(rewards):
    total_plays = rewards[0][1] + rewards[1][1]
    return max([0,1],
               key = lambda i: rewards[i][0] + sqrt(2*log(total_plays+1)/(rewards[i][1] + 1e-6)))

def best_arms(bandits):
    best = dict()
    for s,r in bandits.items():
        best[s] = max([0,1], key= lambda i: r[i][0]/(r[i][1]+1e-6))
    return best


def play_bandit(bandits, state):
    if state[0] > 32:
        return 0
    rewards = bandits.get(state, ((0,0), (0,0)))
    hit = ucb1(rewards)
    if hit:
        card = deal()
        new_state = (state[0]+card, state[1]+(card==11), state[2])
        reward = play_bandit(bandits, new_state)
        bandits[state] = (rewards[0], (rewards[1][0]+reward, rewards[1][1]+1))
    else:
        reward = result(state)
        bandits[state] = ((rewards[0][0]+reward, rewards[0][1]+1), rewards[1])
    return reward

trials = int(5e6)
print('Running for {:,} trials'.format(trials))
res = []
# ucb
for _ in tqdm(range(trials)):
    reward = play_bandit(bandits, random.choice(states))
    res.append(reward)

outfile = 'moves.txt'
outfile2 = 'moves.pickle'
with open(outfile, 'w') as f:
    f.write(str(best_arms(bandits)))
with open(outfile2, 'wb') as f:
    pickle.dump(best_arms(bandits), f)
print('Written results to {}'.format(outfile))

wd = 100
avg_res = [sum(res[i:i+wd])/wd for i in range(len(res)-wd)]
plt.plot(list(range(len(avg_res))), avg_res)
plt.show()

print("Total average win rate over {} trials: {}".format(trials, sum(res)/len(res)))
print("Average win rate over first 1000 trials: {}".format(sum(res[:1000])/len(res[:1000])))
print("Average win rate over last 1000 trials: {}".format(sum(res[-1000:])/len(res[-1000:])))
