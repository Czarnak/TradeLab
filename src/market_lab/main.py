"""Market-Lab GUI entry point."""

from __future__ import annotations

import sys


def main() -> None:
    """Launch the Market-Lab desktop application."""
    from market_lab.utils.config import ensure_dirs
    from market_lab.utils.logging import setup_logging

    setup_logging()
    ensure_dirs()

    # Import strategies so they auto-register
    import market_lab.strategies.ma_crossover  # noqa: F401
    import market_lab.strategies.mean_reversion  # noqa: F401

    from PySide6.QtWidgets import QApplication
    from market_lab.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Market-Lab")
    app.setOrganizationName("MarketLab")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
