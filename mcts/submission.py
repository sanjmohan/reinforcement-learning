# https://www.kaggle.com/c/connectx/overview
# Everything encapsulated in one method because of submission format

import numpy as np

def agent(obs, config):

    def check_outcome(board, col, col_rows):
        # check the spot at (col_rows[col] + 1, col) for winning move, or if col_rows < 0 then draw
        # return None if not game over
        raise NotImplementedError()

    def play(board, col, col_rows, mark):
        row = col_rows[col]
        board[row*config.columns + col] = mark
        col_rows[col] -= 1

    def uct(nodes):
        raise NotImplementedError()

    class MCTSNode():
        def __init__(self, col, mark):
            self.col = col
            self.mark = mark
            self.plays = 0
            self.wins = 0
            self.children = []

        def play(self, board, col_rows):
            play(board, self.col, col_rows, self.mark)

        def expand(self, board, col_rows):
            self.play(board, col_rows)
            if len(children) == config.columns:
                child = ucb(children)
                res = child.expand(board, col_rows)
            else:
                children.append(MCTSNode(len(children)+1, 2-self.mark))
                res = children[-1].simulate(board, col_rows)
            self.plays += 1
            self.wins += 1 - res
            return 1 - res

        def simulate(self, board, col_rows):
            self.play(board, col_rows)
            res = check_outcome(board, self.col, col_rows)
            while res is None:
                # switch player
                mark = 3 - mark
                col = np.random.choice(np.arange(config.columns)[col_rows >= 0])
                #col = np.random.choice(config.columns, p=open_cols)
                play(board, col, col_rows, mark)
                res = check_outcome(board, col, col_rows)
                #if col_rows[col] < 0:
                    #open_cols[col] = 0
            return res

#     # first open row in each column (negative => column is full)
#     col_rows = -1*np.ones(config.columns, dtype=np.int)
#     for c in range(config.columns):
#         for r in range(config.rows):
#             if board[r*c + c] != 0:
#                 break
#             col_rows[c] = r
#     # mask of cols that aren't full
#     #open_cols = np.ones(config.columns, dtype=np.uint8)
#     #for c in range(config.columns):
#     #    if col_rows[c] < 0:
#     #        open_cols[c] = 0

    raise NotImplementedError()

