"""The Game Controller widget

This is the main object. It contains the current Game, and several
State Widgets.
"""

from __future__ import division

import math
from pymt import *

from touchgames.statewidget import StateWidget

widgetPositions = (
        (1 / 4, 0, 0),
        (3 / 4, 0, 0),
        (0, 1 / 2, 1),
        (1 / 4, 1, 2),
        (3 / 4, 1, 2),
        (1, 1 / 2, 3),
    )

def menuItem(name, color=(0, 0, 0)):
    def wrap(func):
        func.name = name
        func.color = color
        return func
    return wrap

class GameController(MTWidget):
    """A game controller.

    Provides selecting a game and displaying common state widgets.
    """
    def __init__(self, game=None, **kwargs):
        MTWidget.__init__(self, **kwargs)
        self.game = None
        self.stateWidgets = []
        self.state = None
        self.paused = False
        self.pauseAnimation = 0
        if game:
            _s = self.start
            self.start = lambda: (_s(), self.startGame(game))

    def startGame(self, game):
        """Start a game, removing the previous one if any"""
        if self.game:
            self.remove_widget(self.game)
        self.game = game
        game.color = self.getColor
        self.add_widget(game, front=False)

    def on_update(self):
        MTWidget.on_update(self)
        dt = getFrameDt()
        # Start self
        self.start()
        self.start = lambda: None
        # Take care of pause animation
        if self.paused:
            if self.pauseAnimation < 1:
                self.pauseAnimation += dt * 3
            if self.pauseAnimation > 1:
                self.pauseAnimation = 1
        else:
            if self.pauseAnimation > 0:
                self.pauseAnimation -= dt * 3
            if self.pauseAnimation < 0:
                self.pauseAnimation = 0
        # Start game if not started
        game = self.game
        if not game.started:
            p = self.get_parent_window()
            game.start(p.width, p.height)
            game.started = True
        if not self.paused:
            game.update(dt)
        elif self.pauseAnimation < 1:
            game.update(dt * (1 - self.pauseAnimation))
        # Update state widgets
        for widget in self.stateWidgets:
            widget.update(dt)

    def draw(self):
        if self.state:
            self.state.draw()

    def start(self):
        """Initialize self (once the parent window is known)"""
        p = self.get_parent_window()
        for x, y, rot in widgetPositions:
            w = StateWidget(
                    self,
                    pos=(p.width * x, p.height * y),
                    rotation=math.pi * rot / 2,
                )
            self.stateWidgets.append(w)
            self.add_widget(w)

    def drawStateContents(self, *args):
        """Pass on a state widget's request for drawing"""
        if self.game:
            self.game.drawStateContents(*args)

    def on_touch_down(self, touch):
        """Touch down event"""
        for widget in self.stateWidgets:
            if widget.on_touch_down(touch):
                return
        if self.game and not self.paused:
            self.game.touchDown(touch)
        return self.paused

    def on_touch_move(self, touch):
        """Touch move event"""
        for widget in self.stateWidgets:
            if widget.on_touch_move(touch):
                return
        MTWidget.on_touch_move(self, touch)
        if self.game and not self.paused:
            self.game.touchMove(touch)
        return self.paused

    def on_touch_up(self, touch):
        """Touch up event"""
        for widget in self.stateWidgets:
            if widget.on_touch_up(touch):
                return
        MTWidget.on_touch_up(self, touch)
        if self.game and not self.paused:
            self.game.touchUp(touch)
        return self.paused

    @property
    def menuContents(self):
        """Retrieve menu contents to be displayed in a state widget"""
        if self.paused:
            return self.game.menuContents(True) + [self.quit, self.unpause]
        else:
            return [self.pause] + self.game.menuContents(False)

    def getColor(self, *rgb):
        """Get a color"""
        if self.paused:
            try:
                r, g, b = rgb
                gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
                p = self.pauseAnimation
                if self.pauseAnimation < 1:
                    return tuple(c * (1 - p) + gray * p for c in rgb)
                else:
                    return (gray, ) * 3
            except ValueError:
                r, g, b, a = rgb
                return self.getColor(r, g, b) + (a,)
        else:
            return rgb

    @menuItem("Pause")
    def pause(self):
        """Pause menu item"""
        self.paused = True

    @menuItem("Resume")
    def unpause(self):
        """Resume menu item"""
        self.paused = False

    @menuItem("Quit")
    def quit(self):
        """Quit menu item"""
        exit()


def registerPyMTPlugin(cls, globals=None):
    """Provides functions required by PyMT's game plugin launcher, and a main()

    cls is the Game class to register.

    Returns pymt_plugin_activate, pymt_plugin_deactivate, main.

    globals is a dictionary. If given, the functions are entered into it.
    Pass the globals() dictionary of a module to register the plugin
    """
    def pymt_plugin_activate(w, ctx):
        ctx.c = GameController(cls())
        w.add_widget(ctx.c)

    def pymt_plugin_deactivate(w, ctx):
        w.remove_widget(ctx.c)

    def main():
        w = MTWindow()
        ctx = MTContext()
        pymt_plugin_activate(w, ctx)
        runTouchApp()
        pymt_plugin_deactivate(w, ctx)

    if globals is not None:
        globals.update(locals())

    return pymt_plugin_activate, pymt_plugin_deactivate, main
