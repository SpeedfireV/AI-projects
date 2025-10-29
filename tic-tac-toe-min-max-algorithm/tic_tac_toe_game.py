from dataclasses import dataclass, replace, field
from enum import Enum

class GameState(Enum):
    Won = 1
    Tie = 0
    Lost = -1
    Ongoing = None

@dataclass(frozen=True)
class TicTacToe:
    board: tuple[str] = None
    player: str = None

    def __post_init__(self):
        # Sets empty board
        if self.board is None:
            object.__setattr__(self, "board", tuple("" for _ in range(9)))
        if self.player is None:
            object.__setattr__(self, "player", "X")

    def moves_left(self) -> list[int]:
        moves: list[int] = []
        for move, value in enumerate(self.board):
            if value == "":
                moves.append(move)
        return moves

    def __is_move_available(self, pos: int):
        return self.board[pos] == ""

    def make_move(self, pos: int):
        if not self.__is_move_available(pos):
            return None
        new_board: tuple[str] = self.board[:pos] + (self.player,) + self.board[pos + 1:]
        return replace(self, board=new_board, player="X" if self.player=="O" else "O")

    def did_player_win(self, player: str) -> GameState:

        """
        Function that checks if someone wins the game with the current state of the board
        """

        terminal_value = lambda winner_sign: GameState.Won if winner_sign == player else GameState.Lost

        # Checks rows
        for i in range(3):
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != "":
                return terminal_value(self.board[i * 3])

        # Checks cols
        for i in range(3):
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != "":
                return terminal_value(self.board[i])

        # Checks cross top-left to bottom-right
        if self.board[0] == self.board[4] == self.board[8] != "":
            return terminal_value(self.board[0])

        # Checks cross from top-right to bottom-left
        if self.board[2] == self.board[4] == self.board[6] != "":
            return terminal_value(self.board[2])

        # Checks if whole board is filled -> if it is return tie [0]
        for value in self.board:
            if value == "":
                return GameState.Ongoing
        else:
            return GameState.Tie

    def show_board(self):
        print()
        for row in range(3):
            row_values = [
                self.board[row * 3 + col] if self.board[row * 3 + col] != "" else " "
                for col in range(3)
            ]
            print(" | ".join(row_values))
            if row < 2:
                print("--+---+--")
        print()

if __name__ == "__main__":
    game = TicTacToe()
    game.show_board()
    game = game.make_move(1)
    game.show_board()
    game = game.make_move(2)
    game.show_board()
    game = game.make_move(0)
    game.show_board()
    game = game.make_move(6)
    game.show_board()
    game = game.make_move(4)
    game.show_board()
    game = game.make_move(3)
    game.show_board()
    game = game.make_move(7)
    game.show_board()

    print(game.did_player_win("X"))
