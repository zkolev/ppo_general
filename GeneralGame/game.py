# Implements basic generala game (https://en.wikipedia.org/wiki/Generala)
# This game is suited for a RL routine

from random import randint
from abc import abstractmethod, ABC


class Dice(object):
    def __init__(self):
        self.value = None

    def roll(self):
        self.value = randint(1, 6)

    def get_face(self):
        return self.value


class Position(ABC):
    def __init__(self, id, name, **kwargs):
        self.id = id
        self.name = name
        self.checked = False

    @abstractmethod
    def checkout(self, **kwargs):
        pass

    def reset(self):
        self.checked = False

    def is_checked(self):
        return self.id, self.checked


class MandatoryPosition(Position):
    """
    Basic positions to checkout. There is 1 position for
    each dice face from 1 to 6. The score is determined by the
    number of dices with the required face. If any of the mandatory
    positions is checked with less than 3 faces one time penalty of
    -25 points (per game, not per position) is applied
    """
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def checkout(self, dice_values):
        self.checked = True
        achieved = False

        score = sum([i for i in dice_values if i == self.value])

        if score >= (3 * self.value):
            achieved = True

        return score, achieved


class NofKind(Position):
    """
    Similar to the mandatory position but without the penalty
    and with required exact number of faces
    5 of kind = general/ general = +50 bonus points
    """
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)

        # Required count
        self.value = value

    def checkout(self, dice_values):
        self.checked = True
        achieved = True
        bp = 0
        ds = sorted(set(dice_values), reverse=True)

        for i in ds:
            n = sum([j == i for j in dice_values])

            if n >= self.value:
                # Add bonus for general
                if self.value == 5:
                    bp = 50

                return self.value * i + bp, achieved

            else:
                continue

        return 0, achieved

# Straight
class Straight(Position):
    """
    All dices with consequent face values (regardless of the order)
    """
    def __init__(self, is_small=True, **kwargs):
        super().__init__(**kwargs)

        if is_small:
            self.value = [1, 2, 3, 4, 5]
        else:
            self.value = [2, 3, 4, 5, 6]

    def checkout(self, dice_values):

        self.checked = True

        score = 0
        _dices_sorted = sorted(dice_values)

        if _dices_sorted == self.value:
            score = sum(_dices_sorted)

        return score, True


# Free position
class Chance(Position):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def checkout(self, dice_values):
        self.checked = True
        return sum(dice_values), True


class FullHouse(Position):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def checkout(self, dice_values):

        self.checked = True

        z = []
        score = 0

        for i in set(dice_values):
            n = sum([i == k for k in dice_values])
            z.append(n)

        if len(z) <=2 and min(z) > 1:
            score = sum(dice_values)

        return score, True


class ScoreBoard(object):
    def __init__(self):
        self.positions = [
            MandatoryPosition(id=0, name='Ones', value=1),
            MandatoryPosition(id=1, name='Twos', value=2),
            MandatoryPosition(id=2, name='Threes', value=3),
            MandatoryPosition(id=3, name='Fours', value=4),
            MandatoryPosition(id=4, name='Fives', value=5),
            MandatoryPosition(id=5, name='Sixs', value=6),

            NofKind(id=6, name='TwoOfKind', value=2),
            NofKind(id=7, name='ThreeOfKind', value=3),
            NofKind(id=8, name='FourOfKind', value=4),

            FullHouse(id=9, name='FullHouse'),
            Straight(id=10, name='StraightLow', is_small=True),
            Straight(id=11, name='StraightHigh', is_small=False),
            Chance(id=12, name='Chance'),
            NofKind(id=13, name='General', value=5)

        ]
        self.not_yet_penalized = True
        self.total_score = 0

    def checkout(self, id, dices):

        # Assert that the given position is unchecked
        assert id in self.__get_unchecked_positions()

        # Obtain the dice values in a list
        dice_values = [d.get_face() for d in dices]

        # One time penalty for not fulfilling the
        # mandatory part is applied to the final score
        pos_ix = self.__get_position_ix(position_id=id)
        _score, _achieved = self.positions[pos_ix].checkout(dice_values=dice_values)

        if self.not_yet_penalized and not _achieved:
            self.not_yet_penalized = False
            _score -= 25

        # Update total score
        self.total_score += _score

        # Check if done:
        done = self.__is_done()

        return _score, done

    def get_scoreboard_status(self):
        """Will be called by the """
        return [int(s) for _, s in [k.is_checked() for k in self.positions]]

    def reset_positions(self):
        """
        Used to  restart the game
        """
        self.not_yet_penalized = True
        self.total_score = 0

        # Reset position status
        for p in self.positions:
            p.reset()

    def __get_unchecked_positions(self):
        return [p_id for p_id, s in [k.is_checked() for k in self.positions] if not s]

    def __get_position_ix(self, position_id):
        for _ix, pos in enumerate(self.positions):
            _id, _ = pos.is_checked()
            if _id == position_id:
                return _ix

    def __is_done(self):
        return all([j for _, j in [k.is_checked() for k in self.positions]])


# Main Game Class
class GeneralGame(object):
    def __init__(self, print_state=True):

        self.dices = [Dice() for _ in range(5)]
        self.score_board = ScoreBoard()
        self.game_round = None
        self.remaining_rolls = None
        self.done = 1
        self.round_rolls = 0
        self.penalized = False
        self.total_score = 0
        self.print_state = print_state

    def start_game(self):

        # Reset positions
        self.score_board.reset_positions()

        # Roll dices
        for dice_ in self.dices:
            dice_.roll()

        self.done = 0
        self.game_round = 1
        self.remaining_rolls = 2
        self.total_score = 0

        s = self.__get_game_state()

        return s, None, 0, 0

    def step(self, a, restart_when_done=True):
        '''
        :param a: tuple (bool, int)
        :param restart_when_done: bool indicating if the agent has to restart after game over
        :return (state, (action, argument), reward, done, total_score)

        The game was designed to with the intention to try simplified version
        of auto regressive policy. That is why the action is composite of two values
        '''

        # Unpack, action (0: checkout position, 1: roll_dices)
        action, argument = a

        assert action in [0, 1]

        # There are t
        if action:
            r, done = self.__roll(argument)

        else:
            r, done = self.__checkout(argument)

            # If checkout and done
            # restart game for rollouts
            if done and restart_when_done:
                _ = self.start_game()

        # Get state
        s = self.__get_game_state()

        return s, a, r, done

    def __roll(self, dice_ix):

        assert self.remaining_rolls > 0

        for ix in dice_ix:
            self.dices[ix].roll()

        self.remaining_rolls -= 1

        return 0, False

    def __checkout(self, position_id):

        reward, done = self.score_board.checkout(id=position_id, dices=self.dices)

        if not done:
            self.__reset_remaining_rolls()
            _, _ = self.__roll(list(range(5)))

        return reward, done

    def __get_game_state(self):

        """
        # State Vector will consist of
        # face of the dices (int)
        # checked out positions (int, boolean)
        # remaining_rolls (int)
        """

        state_vector = [self.remaining_rolls]

        state_vector.extend([sum([i == j for j in self.__get_dice_face()]) for i in range(1,7)])
        state_vector.extend(self.score_board.get_scoreboard_status())

        if self.print_state:
            dice_face = '|'.join([str(d) for d in self.__get_dice_face()])
            ch_positions = ', '.join([f"{p.name}: {p.checked}" for p in self.score_board.positions])

            print(f"Remaining rolls: {self.remaining_rolls}\nDices: |{dice_face}|\nPositions: {ch_positions}")

        return state_vector

    def __get_dice_face(self):
        return [d.get_face() for d in self.dices]

    def __reset_remaining_rolls(self):
        self.remaining_rolls=3