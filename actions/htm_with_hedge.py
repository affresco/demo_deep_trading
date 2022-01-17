import enum


class Actions(enum.Enum):
    """
    Class representing potential actions in the context of a hedged option position.

    Skip: Do nothing
    Sell_Perpetual: Hedge by selling the perpetual futures/swap in the same currency of the option.
    Buy_Perpetual: Hedge by buying the perpetual futures/swap in the same currency of the option.

    """

    # Do nothing
    Skip = 0

    # Sell hedging instrument
    Sell_Perpetual = 1

    # Buy hedging instrument
    Buy_Perpetual = 2
