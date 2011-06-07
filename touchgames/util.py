# Encoding: UTF-8

from __future__ import division

import itertools
import math

from kivy.graphics import (Color, Ellipse, Line, Rectangle, Point,
    Rotate, Translate, Scale, PushMatrix, PopMatrix)

def HollowCircle(pos, radius, segments=50):
    """Draw a circle outline with the specified center and radius

    Return the resulting kivy graphics instruction
    """
    points = list(itertools.chain(*tuple((
                pos[0] + math.cos(t / segments * 2 * math.pi) * radius,
                pos[1] + math.sin(t / segments * 2 * math.pi) * radius,
            )
            for t in range(0, segments + 1)
        )))
    return Line(points=points)

def FilledCircle(pos=(0, 0), radius=1):
    """Draw a filled circle with the specified center and radius

    Return the resulting kivy graphics instruction
    """
    return Ellipse(
            pos=(pos[0] - radius, pos[1] - radius),
            size=(radius * 2, radius * 2),
        )
