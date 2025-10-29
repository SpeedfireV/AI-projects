from tic_tac_toe_game import TicTacToe, GameState


def min_max_algorithm(game: TicTacToe, depth: int, max_depth: int, max_player: str = "X", stacktrace: tuple[TicTacToe] = tuple()): # Return tuple(GameState, TicTacToe, tuple[TicTacToe])
    game_state: GameState = game.did_player_win(player=max_player)
    if game_state != GameState.Ongoing:
        return [game_state, game, stacktrace]

    if depth == max_depth:
        return [GameState.Tie, game, stacktrace]
    else:
        moves_left: list[int] = game.moves_left(False)
        # MAX
        if game.player == max_player:
            best_result = GameState.Lost
            best_game_state = game
            best_game_stacktrace = stacktrace
            for move in moves_left:
                game_after_move = game.make_move(move)
                move_result = min_max_algorithm(game_after_move, depth + 1, max_depth, max_player, stacktrace + (game_after_move, ))
                if move_result[0].value >= best_result.value:
                    best_result = move_result[0]
                    best_game_state = move_result[1]
                    best_game_stacktrace = move_result[2]
                if best_result == GameState.Won:
                    break
            return [best_result, best_game_state, best_game_stacktrace]
        else: # MIN
            worst_result = GameState.Won
            best_game_state = game
            best_game_stacktrace = stacktrace
            for move in moves_left:
                game_after_move = game.make_move(move)
                move_result: GameState = min_max_algorithm(game_after_move, depth + 1, max_depth, max_player, stacktrace + (game_after_move, ))
                if move_result[0].value <= worst_result.value:
                    worst_result = move_result[0]
                    best_game_state = move_result[1]
                    best_game_stacktrace = move_result[2]
                if worst_result == GameState.Lost:
                    break

            return [worst_result, best_game_state, best_game_stacktrace]






if __name__ == "__main__":
    game: TicTacToe = TicTacToe(board=(
    "X", "", "",
    "",  "O", "",
    "",  "",  ""
)
,player="X")
    game_result = GameState.Ongoing
    while len(game.moves_left()) > 0 and (game_result == GameState.Ongoing or game_result == GameState.Tie):
        game_result, game, stacktrace = min_max_algorithm(game, depth=0, max_depth=5, max_player="O")
        for game in stacktrace:
            game.show_board()

    print(game_result)



