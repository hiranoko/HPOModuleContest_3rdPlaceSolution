import numpy as np
from aiaccel.util import aiaccel


def main(p: dict) -> float:
    """The Schwefel function of 5 dimensions with global minimum of 0 at x =
    4.209687.

    Args:
        p (Dict[str, float]): A parameter dictionary.

    Returns:
        float: Calculated objective value.
    """
    x = np.array(list(p.values()))
    if (np.abs(x) > 5).any():
        return np.inf
    y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    y = 100 * x - y0
    f: float = 2094.9144363621604 + (-y * np.sin(np.sqrt(np.abs(y)))).sum()
    return float(f * 0.001834960061484341)


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
