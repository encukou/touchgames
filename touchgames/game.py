
from pymt import *

class Game(MTWidget):
    """A multitouch game

    """

    "Set to True after the game is started"
    started = False

    def update(self, dt):
        """Update game state after "dt" time has elapsed"""

    def start(self, width, height):
        """Start the game

        The window dimensions are given in width, height.
        """

    def draw(self):
        """Draw the game state
        """

    def touchDown(self, touch):
        """Touch Down event.

        Works the same as PyMT's on_touch_down, but is only fired if the game
        is not paused.
        """

    def touchMove(self, touch):
        """Touch Move event

        Works the same as PyMT's on_touch_move, but is only fired if the game
        is not paused.
        """

    def touchUp(self, touch):
        """Touch Up event

        Works the same as PyMT's on_touch_up, but is only fired if the game
        is not paused.
        """

    def drawStateContents(self, width, height, *args, **kwargs):
        """Draw the state widget contents.

        The widget has the given width and height:

              ___________________     _
             /                   \    ^
            /                     \   | Height
            -----------*-----------   -
                       `- (0, 0)

            |<------ width ------>|

        The corners are rounded with radius equal to height.

        Ignore args & kwargs; these are for forward compatibility.
        """

    def color(self, *rgb):
        """Returns the color to use instead of rgb

        Always use this function to get colors.
        It allows for a monochrome effect when paused.
        """
        return rgb

    def menuContents(self, paused):
        """Return actions to be used in the game's menu

        Actions have name and color attributes that determine how they're
        drawn. When a user selects an action, it is called like a function.

        The list can be different depending on whether the game is paused.

        The Quit and Pause/Resume actions are supplied by the framework.
        """
        return []
