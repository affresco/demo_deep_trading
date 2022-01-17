import enum


class Actions(enum.Enum):

    # Do nothing
    Skip = 0

    # Sell hedging instrument
    Sell_Perpetual = 1

    # Buy hedging instrument
    Buy_Perpetual = 2
