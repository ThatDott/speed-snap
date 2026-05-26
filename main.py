import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.gui import SpeedRadarGUI


if __name__ == "__main__":
    app = SpeedRadarGUI()
    app.mainloop()
