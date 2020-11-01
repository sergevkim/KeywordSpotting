import string

from torch import Tensor


class TokenConverter:
    ALPHABET = list(' ' + string.ascii_lowercase)
    SYMBOL2NUMBER = dict()
    NUMBER2SYMBOL = dict()

    for number, symbol in enumerate(ALPHABET):
        SYMBOL2NUMBER[symbol] = number
        NUMBER2SYMBOL[number] = symbol

    @classmethod
    def symbols2numbers(cls, symbols):
        result = list(cls.SYMBOL2NUMBER[symbol] for symbol in symbols)

        return result

    @classmethod
    def numbers2symbols(cls, numbers):
        result = ''.join(cls.NUMBER2SYMBOL[number] for number in numbers)

        return result

