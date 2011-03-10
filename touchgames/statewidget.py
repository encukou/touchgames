# Encoding: UTF-8

from __future__ import division

import math
import weakref

from OpenGL.GL import glTranslatef, glRotatef, GL_LINE_LOOP
from pymt import *
from pymt.lib import transformations
import numpy

class StateWidget(MTWidget):
    """A widget that displays game state, and also serves as a drag-to-open menu
    """
    width = 150
    radius = 30
    _stickout = 0
    retract_timer = 0
    menu_spacing = 32

    def __init__(self, controller, rotation=0, **kwargs):
        MTWidget.__init__(self, **kwargs)
        self.rotation = rotation
        self.controller = controller
        self.matrix = numpy.dot(
                transformations.translation_matrix((self.x, self.y, 0)),
                transformations.rotation_matrix(-rotation, (0, 0, 1)),
            )
        self.inv_matrix = numpy.linalg.inv(self.matrix)
        self.touches = weakref.WeakKeyDictionary()

    def draw(self):
        x, y = self.pos
        w = self.width
        r = self.radius
        m = self.stickout_full
        s = self.menu_spacing
        c = self.controller.pauseAnimation
        set_color(.75, .75 + c / 8, .75 + c / 4, 1)
        with gx_matrix:
            glTranslatef(x + 0.01, y + 0.01, 0)
            glRotatef(-self.rotation / math.pi * 180, 0, 0, 1)
            if self.stickout:
                glTranslatef(0, self.stickout, 0)
            drawCircle(pos=(w / 2, 0), radius=r)
            drawCircle(pos=(-w / 2, 0), radius=r)
            drawRectangle(pos=(-w / 2, 0), size=(w, r))
            drawRectangle(pos=(-w / 2 - r, -m), size=(w + r * 2, m))

            self.controller.drawStateContents(w, r)

            if self.stickout:
                for i, menuItem in enumerate(self.menuContents):
                    drawLabel(menuItem.name, pos=(0, (i + 0.5) * -s), font_size=10, center=True, color=(0, 0, 0))

    @property
    def menuContents(self):
        return self.controller.menuContents

    @property
    def stickout_max(self):
        return self.menu_spacing * len(self.menuContents)
    stickout_full = stickout_max

    @property
    def stickout(self):
        a = self.controller.pauseAnimation
        return self._stickout * (1 - a) + len(self.menuContents) * self.menu_spacing * a

    def collide_point(self, x, y):
        x, y, z, w = numpy.dot(self.inv_matrix, (x, y, 0, 1))
        y -= self.stickout
        w = self.width / 2 + self.radius
        if -w < x < w and y < self.radius:
            return x, y
        else:
            return False

    def on_touch_down(self, touch):
        x_y = self.collide_point(touch.x, touch.y)
        if x_y:
            x, y = x_y
            self.retract_timer = 2
            if y > -self.menu_spacing * 0.25:
                touch.grab(self)
                self.touches[touch] = x, y, self.stickout
            else:
                i = - (y - 0.25) / self.menu_spacing
                self.menuContents[int(i)]()
            return True

    def on_touch_move(self, touch):
        try:
            x, y, s = self.touches[touch]
        except KeyError:
            return
        else:
            startx, starty, z, w = numpy.dot(self.inv_matrix, (touch.x, touch.y, 0, 1))
            self._stickout = starty - y + s
            self._stickout = max(self._stickout, 0)
            self._stickout = min(self._stickout, self.stickout_max)

    def update(self, dt):
        if self._stickout > 0 and self.controller.pauseAnimation < 1:
            self.retract_timer -= dt
            if self.retract_timer < 0:
                self._stickout -= dt * 20 * (self.retract_timer ** 2)
        else:
            self._stickout = 0
