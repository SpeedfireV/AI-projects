from dataclasses import dataclass

from tic_tac_toe_game import TicTacToe, GameState


def min_max_algorithm(game: TicTacToe, depth: int, max_depth: int, max_player: str = "X"): # Return tuple(GameState, TicTacToe)
    game_state: GameState = game.did_player_win(player=max_player)
    if game_state != GameState.Ongoing:
        return [game_state, game]
    else:
        moves_left: list[int] = game.moves_left()
        # MAX
        if game.player == max_player:
            best_result = GameState.Lost
            best_game_state = game
            for move in moves_left:
                game_after_move = game.make_move(move)
                move_result: GameState = min_max_algorithm(game_after_move, depth + 1, max_depth, max_player)

                if move_result[0].value >= best_result.value:
                    best_result = move_result[0]
                    best_game_state = move_result[1]
            return [best_result, best_game_state]
        else: # MIN
            worst_result = GameState.Won
            best_game_state = game
            for move in moves_left:
                game_after_move = game.make_move(move)
                move_result: GameState = min_max_algorithm(game_after_move, depth + 1, max_depth, max_player)
                if move_result[0].value <= worst_result.value:
                    worst_result = move_result[0]
                    best_game_state = move_result[1]

            return [worst_result, best_game_state]






if __name__ == "__main__":
    game: TicTacToe = TicTacToe(board=(
    "", "", "",
    "",  "X", "O",
    "",  "",  ""
), player="O")
    game_result, terminal_game_state = min_max_algorithm(game, depth=1, max_depth=3, max_player="O")
    print(game_result)
    terminal_game_state.show_board()
    # game.show_board()
    # game = game.make_move(5)
    # game.show_board()
    # game = game.make_move(6)
    # game.show_board()



