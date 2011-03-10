
from __future__ import division

import random
import gzip
import cPickle as pickle
from collections import defaultdict

from touchgames.game import Game
from touchgames.gamecontroller import registerPyMTPlugin

from touchgames.maze import Maze

from pymt import *
from OpenGL.GL import *
from gillcup.animatedobject import AnimatedObject
from gillcup.timer import Timer

class ReplayAdapter(Game):
    def __init__(self, replayFile, **kwargs):
        # Initialize
        print replayFile
        kwargs['log_to'] = None
        Game.__init__(self, **kwargs)
        if isinstance(replayFile, basestring):
            replayFile = open(replayFile, 'rb')
        self.ghosts = {}
        self.replayFile = gzip.GzipFile(fileobj=replayFile, mode='rb')
        self.replayEvent()

    def draw(self):
        self.game.draw()
        self.ghosts = dict((k, v) for k, v in self.ghosts.items() if v.draw(self))

    def drawStateContents(self, width, height, *args, **kwargs):
        self.game.drawStateContents(width, height, *args, **kwargs)

    def _log(self, *args):
        pass

    def replayEvent(self):
        try:
            event = pickle.load(self.replayFile)
        except EOFError:
            print 'Replay end'
            return
        time, eventtype = event[:2]
        time /= 1000
        args = event[2:]
        print 'Replaying', event
        func = None
        if eventtype == 'random':
            self.game = Maze(random_state=args[0])
        elif eventtype == 'replay_args':
            self.game.start(**args[0])
        elif eventtype == 'update':
            dt = args[0] / 1000
            func = lambda: (self.game.timer.advance(dt), self.game.update(dt))
        elif eventtype == 'touch':
            tp, touch = args
            touch = FakeTouch(touch)
            if tp == 'down':
                self.game.touchDown(touch)
                self.ghosts[touch.id] = Ghost(touch)
            elif tp == 'move':
                self.game.touchMove(touch)
                try:
                    self.ghosts[touch.id].addPoint(touch.x, touch.y)
                except KeyError:
                    pass
            elif tp == 'up':
                self.game.touchUp(touch)
                try:
                    self.ghosts[touch.id].animate('opacity', 0, time=1, timer=self.timer)
                except KeyError:
                    pass
            else:
                raise AssertionError('Unknown touch type %s' % tp)
        else:
            raise AssertionError('Unknown event type %s' % eventtype)
        if func:
            self.timer.schedule(time, func)
        self.timer.schedule(time, self.replayEvent)

class Ghost(AnimatedObject):
    def __init__(self, touch=None):
        AnimatedObject.__init__(self)
        self.points = []
        self.opacity = 1
        if touch:
            self.addPoint(touch.x, touch.y)

    def addPoint(self, *point):
        self.points.append(point)

    def draw(self, controller):
        if self.opacity > .5:
            bo = 1
            wo = (self.opacity - .5) * 2
        else:
            bo = self.opacity * 2
            wo = 0
        try:
            lx, ly = self.points[-1]
        except IndexError:
            return
        rect = [
                    (lx-20, ly-20),
                    (lx-20, ly+20),
                    (lx+20, ly+20),
                    (lx+20, ly-20),
                ]
        controller.set_color(0.2, 0.2, 0.2, bo)
        drawPolygon(self.points, GL_LINE_STRIP, linewidth=3)
        controller.set_color(0.8, 0.8, 0.8, wo)
        drawPolygon(self.points, GL_LINE_STRIP, linewidth=1)
        if self.opacity == 1:
            controller.set_color(0, 0, 0, bo)
            drawPolygon(rect, GL_LINE_LOOP, linewidth=4)
            controller.set_color(1, 1, 1, wo)
            drawPolygon(rect, GL_LINE_LOOP, linewidth=2)
        return self.opacity

class FakeTouch(object):
    def __init__(self, attrs):
        for name, value in attrs.items():
            setattr(self, name, value)

registerPyMTPlugin(ReplayAdapter, globals())

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
