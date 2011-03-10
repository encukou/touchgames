
import random
import cPickle as pickle
import gzip

from pymt import *
from OpenGL.GL import *
from gillcup.animatedobject import AnimatedObject
from gillcup.timer import Timer

class Game(MTWidget, AnimatedObject):
    """A multitouch game

    """

    "Set to True after the game is started"
    started = False

    def __init__(self, log_to=None, random_state=None, **kwargs):
        # Get our own source of randomness
        self.random = random.Random()
        if random_state:
            self.random.setstate(random_state)
        else:
            self.random.jumpahead(random.randint(1, 10000))
        if isinstance(log_to, basestring):
            log_to = open(log_to, 'wb')
        if log_to:
            self.__log_fileobj = log_to
            self.__log_file = gzip.GzipFile(fileobj=log_to)
        # Initialize
        MTWidget.__init__(self, **kwargs)
        AnimatedObject.__init__(self, Timer())
        if log_to:
            self.__log_lastTime = self.__log_lastFlush = self.timer.time
            self._log('random', self.random.getstate())

    def update(self, dt):
        """Update game state after "dt" time has elapsed"""

    def start(self, width, height, **args):
        """Start the game

        The window dimensions are given in width, height; if a dict is returned,
        it will be given as **args when replayed.
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

    def set_color(self, *rgb):
        if len(rgb) == 3:
            glColor3f(*self.color(*rgb))
        else:
            glColor4f(*self.color(*rgb))

    def menuContents(self, paused):
        """Return actions to be used in the game's menu

        Actions have name and color attributes that determine how they're
        drawn. When a user selects an action, it is called like a function.

        The list can be different depending on whether the game is paused.

        The Quit and Pause/Resume actions are supplied by the framework.
        """
        return []

    def log(self, eventtype, *args):
        """Log a custom event. event is a string

        When the game is replayed replay(eventtype, *args) will be called
        at the appropriate time.
        """
        self._log('custom', eventtype, *args)

    def _log(self, *args):
        """Log an event"""
        if self.__log_file:
            entry = (int((self.timer.time - self.__log_lastTime) * 1000), ) + args
            pickle.dump(entry, self.__log_file, protocol=2)
            self.__log_lastTime = self.timer.time
            if self.timer.time - self.__log_lastFlush > 1:
                self.__log_file.flush()
                self.__log_fileobj.flush()
                self.__log_lastFlush = self.timer.time

    def logTouch(self, eventtype, touch):
        self._log('touch', eventtype, dict(
                x=touch.x, y=touch.y, id='replay-%s' % touch.id,
            ))

    def replay(self, eventtype, *args):
        """Call replay_<eventtype>(*args), if the method exists"""
        try:
            func = getattr(self, 'replay_' + eventtype)
        except AttributeError:
            pass
        else:
            func(*args)
