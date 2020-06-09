# https://www.kaggle.com/c/connectx/overview
# Everything encapsulated in one method because of submission format

def agent(obs, config):
    import time
    start_time = time.time()
    import numpy as np

    # constants determined from game size
    col_inds = np.arange(config.columns)
    allmask = (1 << (2*config.rows*config.columns)) - 1
    left_col_mask = 0
    right_col_mask = 0
    for _ in range(config.rows):
        left_col_mask = (left_col_mask << 2*config.columns) ^ (0b11 << 2*(config.columns-1))
        right_col_mask = (right_col_mask << 2*config.columns) ^ 0b11
    masks = [~left_col_mask & allmask, ~right_col_mask & allmask, allmask, ~left_col_mask & allmask]
    directions = list(zip([1, config.columns-1, config.columns, config.columns+1], masks))

    def check_filled(col_rows):
        return np.all(col_rows < 0)

    def check_won(board, col_rows):
        # row win, diag win, column win, diag win
        for direction, mask in directions:
            dir_win = board
            shift = board
            for _ in range(config.inarow - 1):
                shift = (shift >> (direction*2)) & mask
                dir_win = dir_win & shift
            if dir_win:
                return True
        return False

    def play(board, col, col_rows, mark):
        row = col_rows[col]
        top = config.columns*config.rows*2 - 2
        board = board | (mark << (top-2*int(row*7 + col)))
        col_rows[col] -= 1
        return board

    def uct(nodes, t, c=1):
        best = np.argmax([n.wins/n.plays + c*np.sqrt(2*np.log(t)/n.plays) for n in nodes])
        return nodes[best]

    class MCTSNode():
        def __init__(self, col, mark):
            self.col = col
            self.mark = mark
            self.plays = 0
            self.wins = 0.0
            self.children = []

        def expand(self, board, col_rows):
            board = play(board, self.col, col_rows, self.mark)
            self.plays += 1
            if len(self.children) == 0:
                if check_won(board, col_rows):
                    self.wins += 1.0
                    return 1
                if check_filled(col_rows):
                    self.wins += 0.5
                    return 0.5
            if len(self.children) == config.columns:
                # use upper confidence bound to choose best move out of valid moves
                valid_moves = [ch for col,ch in enumerate(self.children) if col_rows[col] >= 0]
                child = uct(valid_moves, self.plays)
                child_res = child.expand(board, col_rows)
            else:
                self.children.append(MCTSNode(len(self.children), 3-self.mark))
                child_res = self.children[-1].simulate(board, col_rows)
            self.wins += 1 - child_res
            return 1 - child_res

        def simulate(self, board, col_rows):
            board = play(board, self.col, col_rows, self.mark)
            self.plays += 1
            mark = self.mark
            while not check_won(board, col_rows):
                if check_filled(col_rows):
                    return 0.5
                # switch player
                mark = 3 - mark
                col = np.random.choice(col_inds[col_rows >= 0])
                board = play(board, col, col_rows, mark)
            res = int(mark == self.mark)
            self.wins += res
            return res

    board = 0
    for b_ in obs.board:
        board = (board << 2) | b_
    col_rows = -1*np.ones(config.columns, dtype=np.int)
    for c in range(config.columns):
        for r in range(config.rows):
            if obs.board[r*config.columns + c] != 0:
                break
            col_rows[c] = r
    # Initialize nodes
    nodes = [MCTSNode(c, obs.mark) for c in range(config.columns) if col_rows[c] >= 0]
    # allocate space outside of loop so the loop can reuse the space
    tmpcol_row = np.copy(col_rows)
    for node in nodes:
        node.expand(board, tmpcol_row)
        # reset board info
        np.copyto(tmpcol_row, col_rows)
    # Proceed by UCT
    t = 0
    while time.time() - start_time < config.timeout - 0.15:
        node = uct(nodes, t+1)
        node.expand(board, tmpcol_row)
        np.copyto(tmpcol_row, col_rows)
        t += 1
    # Choose best move
    best = int(np.argmax([n.wins/n.plays for n in nodes]))
    return nodes[best].col
