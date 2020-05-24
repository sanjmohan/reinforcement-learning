# https://www.kaggle.com/c/connectx/overview
# Everything encapsulated in one method because of submission format

def agent(obs, config):
    import numpy as np

    def check_outcome(board, col, col_rows):
        # 1 = win (for mark at spot (col_rows[col], col)), 0 = draw, None = not finished
        # Check if board filled:
        for c in range(config.columns):
            if col_rows[c] >= 0:
                break
        else:
            return 0  # draw
        row = col_rows[col] + 1
        mark = board[row, col]
        # check column - will only be a streak in the top `inarow` spots of the column
        if row < config.rows - config.inarow + 1:
            for r in range(config.inarow):
                if board[row + r, col] != mark:
                    break
            else:
                return 1  # column win
        # check row + diags
        row_matches = 0
        updiag_matches = 0
        downdiag_matches = 0
        for c in range(max(0, col-config.inarow+1), min(config.columns, col+config.inarow)):
            offset = c - col
            if board[row, c] == mark:
                row_matches += 1
                if row_matches == config.inarow:
                    return 1  # row win
            else:
                row_matches = 0
            if 0 <= row + offset < config.rows and board[row+offset, c] == mark:
                updiag_matches += 1
                if updiag_matches == config.inarow:
                    return 1  # upwards diagonal win
            else:
                updiag_matches = 0
            if 0 <= row - offset < config.rows and board[row-offset, c] == mark:
                downdiag_matches += 1
                if downdiag_matches == config.inarow:
                    return 1  # downward diagonal win
            else:
                downdiag_matches = 0
        return None
    
    # in place
    def play(board, col, col_rows, mark):
        row = col_rows[col]
        board[row, col] = mark
        col_rows[col] -= 1
        
    def uct(nodes, t, c=1):
        try:
            best = np.argmax([n.wins/n.plays + c*np.sqrt(2*np.log(t)/n.plays) for n in nodes])
        except:
            print([n.__dict__ for n in nodes])
        return nodes[best]

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
            self.plays += 1
            if len(self.children) == 0:
                this_res = check_outcome(board, self.col, col_rows)
                if this_res is not None:
                    self.wins += this_res
                    return this_res
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
            self.play(board, col_rows)
            self.plays += 1
            mark = self.mark
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
            if res == 1:
                if mark == self.mark:
                    self.wins += 1
                    return 1
                return 0
            return 0.5

    board = np.copy(obs.board)
    board = board.reshape((config.rows, config.columns))
    col_rows = -1*np.ones(config.columns, dtype=np.int)
    for c in range(config.columns):
        for r in range(config.rows):
            if board[r,c] != 0:
                break
            col_rows[c] = r
    # Initialize nodes
    nodes = [MCTSNode(c, obs.mark) for c in range(config.columns) if col_rows[c] >= 0]
    for node in nodes:
        tmpboard = np.copy(board)
        tmpcol_row = np.copy(col_rows)
        node.expand(tmpboard, tmpcol_row)
    # Proceed by UCT
    num_iters = 10000
    for t in range(num_iters):
        tmpboard = np.copy(board)
        tmpcol_row = np.copy(col_rows)
        node = uct(nodes, t+1)
        node.expand(tmpboard, tmpcol_row)
    # Choose best move
    best = int(np.argmax([n.wins/n.plays for n in nodes]))
    return nodes[best].col
#     print( int(np.argmax([n.wins/n.plays for n in nodes])))
#     return nodes