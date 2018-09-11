#!/usr/bin/env python

import numpy as np
import sys


table = np.zeros((6, 7), np.int8)
# Choose an ordering focused on middle of board in order to
# speed up alpha beta pruning
order = np.array([3, 2, 4, 1, 5, 0, 6])


# Get groups of four for a connect four board representing all
# groups of slots that can be used to win a game
def get_fours(board):
	fours = []
	for row in range(board.shape[0]):
		for col in range(board.shape[1] - 3):
			fours.append(board[row, col : col + 4])
	for row in range(board.shape[0] - 3):
		for col in range(board.shape[1]):
			fours.append(board[row : row + 4, col])
	diags = [board.diagonal(i) for i in range(-2, 4)]
	diags.extend(board[::-1, :].diagonal(i) for i in range(-2, 4))
	for diag in diags:
		for i in range(len(diag) - 3):
			fours.append(diag[i : i + 4])
	return fours


# Check whether the current game has finished
def game_over(board, fours):
	if np.count_nonzero(board) == board.size:
		return True, 0

	for four in fours:
		if np.sum(four) == -4:
			return True, -1
		elif np.sum(four) == 4:
			return True, 1
	return False, 0


# Run heuristic function, using different possible heuristic sizes
def heuristic(heuristic_type, fours):
	score = 0
	if heuristic_type == 1:
		sample = np.random.choice(len(fours), 10)
		for i in sample:
			score += np.sum(fours[i])
	elif heuristic_type == 2:
		for four in fours:
			score += np.sum(four) ** 3
	else:
		for four in fours:
			if not(-1 in four and 1 in four) and np.sum(four) != 0:
				score += 10 ** (np.abs(np.sum(four)) - 1) * np.sign(np.sum(four))
	return score


# Check which moves are possible
def valid_moves(board):
	global order
	moves = []
	for col in order:
		z = np.where(board[:, col] == 0)[0]
		if len(z) != 0:
			moves.append((z[-1], col))
	return moves


# Make a move on the board
def move(board, col, player):
	board[valid_moves(board)[col][0], valid_moves(board)[col][1]] = player


# Run max node for a maximizer
def max_node(board, depth, alpha, beta, player, fours, heuristic_type):
	over, winner = game_over(board, fours)
	if over and winner == 0:
		return 0
	if over or depth == 0:
		score = heuristic(heuristic_type, fours)
		return score * player

	v = float('-inf')
	for row, col in valid_moves(board):
		board[row, col] = player
		v = max(v, min_node(board, depth - 1, alpha, beta,
							player, get_fours(board), heuristic_type))
		board[row, col] = 0
		if v >= beta:
			return v
		alpha = max(v, alpha)
	return v


# Run min node for a minimizer
def min_node(board, depth, alpha, beta, player, fours, heuristic_type):
	over, winner = game_over(board, fours)
	if over and winner == 0:
		return 0
	if over or depth == 0:
		score = heuristic(heuristic_type, fours)
		return score * -player

	v = float('inf')
	for row, col in valid_moves(board):
		board[row, col] = -player
		v = min(v, max_node(board, depth - 1, alpha, beta,
							player, get_fours(board), heuristic_type))
		board[row, col] = 0
		if v <= alpha:
			return v
		beta = min(v, beta)
	return v


# Run alpha beta pruning tree
def alphabeta_pruning(board, depth, player, fours, heuristic_type):
	global order
	values = []
	v = float('-inf')
	for row, col in valid_moves(board):
		board[row, col] = player
		v = max(v, min_node(board, depth - 1, float('-inf'), float('inf'),
							player, fours, heuristic_type))
		values.append(v)
		board[row, col] = 0
	if len(values) == 0:
		return -1, 0
	best_score = max(values)
	best_col = values.index(best_score)
	return best_col, best_score


# Runs alpha beta pruning tree search with 2 agents. First player uses heuristic1
# and second player uses heuristic2. Player1 is run for depths from depth1_start to
# depth1_end, against all depths for player 2 (similar). These games are played
# a total or num_times.
#
# Prints a table showing wins and losses for the players, with rows representing
# different depths for player 1 (from depth1_start to depth1_end) and columns
# representing different depths for player 2 (from depth2_start to depth2_end)
def main(heuristic1, depth1_start, depth1_end,
		 heuristic2, depth2_start, depth2_end, num_times):
	depths = np.zeros((depth1_end - depth1_start, depth2_end - depth2_start), np.int8)
	for _ in range(num_times):
		for depth1 in range(depth1_start, depth1_end):
			for depth2 in range(depth2_start, depth2_end):
				board = np.copy(table)
				fours = get_fours(board)
				while True:
					col, _= alphabeta_pruning(board, depth1, -1, fours, heuristic1)
					over, winner = game_over(board, fours)
					if over:
						depths[depth1 - depth1_start][depth2 - depth2_start] += winner
						break
					else:
						move(board, col, -1)

					fours = get_fours(board)
					col, _ = alphabeta_pruning(board, depth2, 1, fours, heuristic2)
					if over:
						depths[depth1 - depth1_start][depth2 - depth2_start] += winner
						break
					else:
						move(board, col, 1)

					fours = get_fours(board)

	print(depths)


# Arguments:
# sys.argv[1] = heuristic1
# sys.argv[2] = depth1_start
# sys.argv[3] = depth1_end
# sys.argv[4] = heuristic2
# sys.argv[5] = depth2_start
# sys.argv[6] = depth2_end
# sys.argv[7] = num_times
if __name__ == '__main__':
	main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
		 int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
