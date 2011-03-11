#! /usr/bin/env python
# Encoding: UTF-8

from __future__ import division


import numpy as np
from numpy.random import random_integers as rnd
import matplotlib.pyplot as plt

from gillcup.animatedobject import AnimatedObject
from gillcup import easing

def maze(width=81, height=51, complexity=0.75, density=0.75):
    # http://en.wikipedia.org/wiki/Maze_generation_algorithm
    # Only odd shapes
    shape = ((height//2)*2+1, (width//2)*2+1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity*(5*(shape[0]+shape[1])))
    density    = int(density*(shape[0]//2*shape[1]//2))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0,:] = Z[-1,:] = 1
    Z[:,0] = Z[:,-1] = 1
    # Make isles
    for i in range(density):
        x, y = rnd(0,shape[1]//2)*2, rnd(0,shape[0]//2)*2
        Z[y,x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:           neighbours.append( (y,x-2) )
            if x < shape[1]-2:  neighbours.append( (y,x+2) )
            if y > 1:           neighbours.append( (y-2,x) )
            if y < shape[0]-2:  neighbours.append( (y+2,x) )
            if len(neighbours):
                y_,x_ = neighbours[rnd(0,len(neighbours)-1)]
                if Z[y_,x_] == 0:
                    Z[y_,x_] = 1
                    Z[y_+(y-y_)//2, x_+(x-x_)//2] = 1
                    x, y = x_, y_
    return Z

import sys
import random
import heapq

import numpy

from pymt import *
from OpenGL.GL import *

import touchgames
from touchgames.game import Game
from touchgames.gamecontroller import registerPyMTPlugin

from touchgames.games.maze.mazesolver import solvemaze

gdb = GestureDatabase()
for build, gesture in (
        (True, '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'),
        (False, 'eNqllttqHEcURd/7R6SXiHO//IDzGtAHBMcWxsSJBkmG+O+zqzzqbkOSEkQIBma6dp86a5/L7efLtz9e7j49PL98fXrYfr5+Xmi7/Xjh7f7m+eXp8feH55vtItvtl4tut/944n4+tl1snHOcuzx+/vNlHItxLP/l2C/jqe1S41Tj1DccYNre0Z0Em4pIumoL5fZ8f/PX+JW3dz/RHbWyZVMkpUil6Pb82/v/fAnLvJNun17fkFTc5mRFZgWFT1d1fFsmJdyIIrxjLT4vzr6LXwMXs+a04kNcmqqKuJPYs8rX4jHF81Wc3ajS2zLcvfkkrmGqppZajltkrsVn8rlfxakCEXYZro30utmhbqRVotH4mQL5X6rLNJLwd/UBbqRdNTwqjCFzCh65IiSGiZAb3GsdvEymors8zrJ1tGoSqZ3FI0haQkIbzhrvXYlPpuK7OA6VWYwMmSR7nNSTI6sRvat76NoxMqFKHuoNM0AhBkO8x/9X7JOq9K4+wJVLwc2kqXISl4A/xYiMPLhrKa4Tqh5QTZtV0kzgdXD7wZBXs1SjZBtFuFSfTPVg6hQtrY6sRJfiakctpTZuFoCBJhFvSIxOqHpAdYORUxPSXv6DZbiaETSaA+rJON6gPqHqAdUL2YbX0cHg6D6l/UiJgm5yvUF9QtU+qcMXrt1ehhZ4LtXX/tKEemVBGEt5m1iNl/KjSSRMSq5gju4G56/VJ1Y7YQ2F9VwarhR423d1IRRBGvwo47W+zoxNqnai6sKS1Ggp3ZkRh3gHcMAqGYCasTakTah2gnqNGb7QMTtyF9dizA0UMHojAqA3iE+mdjC1rChCxMXoBNxH5OgxaL7oDLASUpTrQvVJ1E+FyoGxVIlPaoyHg6hzkjBAF5zYzusG5hOoH0AxkUsLZhlj2XOv05EwlC3DkoViM6RnLf59kTj1Xq5sVClzBDROgcM7yQqbq6CaZO1Enzz94MmY1lZXPyoPideUV2UKHCSGG+kbzOKTpx88MTKJUH8ocbTxqDpChzlhok4kpTAUfR16TKDB+7B2wZIRCWBowBx2hK5olZkH1jXQmEBjX5B47CnoitiAxNEk9dB2TJM2VBcyx3N1WmlPnrHvR4Mm+i1mg5OigupUn0zo8di/HPsfpspae/KMfT0SpzGNGp8eKYcyo5tHaZM7YpY1zJgwY9+NBJMM86Cxe5Hx7HpXbUZdQhcUUKSR64TkJJk7SbQUHQ0Ve2/BJnr0FDI0GthSQ3msY7xuhzlJ5k5SsVEVveaF8rTq4pWYS0g3aANr9Mz3+L+/eXn88vD0/s8PD9i70SqmJX7847GWX2P49fL0+PHrh5fxsG/vCtI06xVZb+x5w913fwPv4lI1'),
    ):
    g = gdb.str_to_gesture(gesture)
    g.build = build
    g = gdb.add_gesture(g)

class Ball(AnimatedObject):
    radius = 0.4

    def __init__(self, maze, coord):
        super(Ball, self).__init__(timer=maze.timer)
        self.maze = maze
        self.coord = numpy.array(coord) + 0.5
        self.target = self.coord
        self._touched = False
        self.blockRadius = 2

    def getDrawCoord(self, coord):
        return coord * (
                self.maze.cell_width,
                self.maze.cell_height,
            )

    def draw(self):
        drawCoord = self.getDrawCoord(self.coord)
        drawRadius = self.radius * self.maze.cell_size
        self.maze.set_color(0.5, 0.5, 0.5, 0.3)
        drawCircle(drawCoord, self.maze.cell_size * self.blockRadius)
        self.maze.set_color(0, 0, 1, 0.1)
        drawCircle(drawCoord, self.maze.cell_size * self.blockRadius, linewidth=3)
        self.maze.set_color(0.5, 0.5, 0.5, 0.3)
        drawCircle(drawCoord, self.maze.cell_size * 2, linewidth=3)
        self.maze.set_color(0, 0, 1)
        drawCircle(drawCoord, drawRadius)

    @property
    def touched(self):
        return self._touched

    @touched.setter
    def touched(self, touched):
        self._touched = touched
        if touched:
            self.animate('blockRadius', 7, time=0.5, easing=easing.quad.out)
        else:
            self.animate('blockRadius', 2, time=1, easing=easing.quart)

    def blocks(self, coord):
        if self.touched:
            return sum((self.coord - coord + 0.5) ** 2) < self.blockRadius ** 2
        else:
            return sum((self.coord - coord + 0.5) ** 2) < self.blockRadius ** 2

    def hittest(self, tileCoord):
        return sum((tileCoord - self.coord) ** 2) <= (
                (self.radius * 5) ** 2
            )

    def update(self, dt):
        distance_covered = 0
        while distance_covered < dt * 10:
            if abs(sum(self.target - self.coord)) < 0.01:
                break
            distance_covered += self.dragBy(self.target - self.coord)
        if (numpy.floor(self.coord) == (0, 1)).all():
            self.maze.balls = [b for b in self.maze.balls if b != self]
            self.maze.win()

    def isOut(self):
        return sum((self.coord -
                (self.maze.width, self.maze.height)) ** 2
            ) > self.maze.home_size ** 2

    def dragBy(self, delta):
        # returns distance covered
        length = numpy.sqrt(sum(delta ** 2))
        max_len = self.radius / 2
        if length > max_len:
            delta = delta / length * max_len
            length = max_len
        radius = self.radius
        for i in (0, 1):
            # Check the up/down & left/right points of the circle
            pt = [0, 0]
            pt[i] = radius
            if self.maze.isWall(self.coord + delta + pt):
                delta[i] = int(self.coord[i] + delta[i] + 1) - radius - self.coord[i]
            if self.maze.isWall(self.coord + delta - pt):
                delta[i] = int(self.coord[i] + delta[i]) + radius - self.coord[i]
        coord = self.coord + delta
        # Get the closest grid intersection
        corner = coord.round()
        # Get a point in the tile behind this corner
        corner_tile = 2 * corner - coord
        # Check that
        if self.maze.isWall(corner_tile):
            vector_to_corner = corner - coord
            if sum(vector_to_corner ** 2) < self.radius ** 2:
                # Part of the ball is inside a corner tile; push it back
                distance_to_corner = numpy.sqrt(sum(vector_to_corner ** 2))
                direction_to_corner = vector_to_corner / distance_to_corner
                coord -= direction_to_corner * (self.radius - distance_to_corner)
                # Penalize this operation
                length += distance_to_corner
        self.coord = coord
        return length

class Flash(AnimatedObject):
    def __init__(self, maze, coord, color=(1,1,0), *args, **kwargs):
        AnimatedObject.__init__(self, maze.timer)
        self.color = color
        self.coord = coord
        self.maze = maze
        self.opacity = 1
        self.radius = 50
        self.setup(*args, **kwargs)

    def setup(self):
        pass

    def draw(self):
        self.maze.set_color(*self.color + (self.opacity, ))
        drawCircle(self.coord, self.radius)
        return self.opacity > 0

    def end(self):
        self.animate('opacity', 0, time=0.5)

class BuildStartFlash(Flash):
    def setup(self, point):
        print 'Build-start flash!', point
        self.opacity = 0
        self.points = list(point)
        self.animate('radius', self.maze.cell_size * 0.7, time=0.5)
        self.animate('opacity', 0.5, time=0.5)

    def explode(self):
        self.animate('radius', 200, time=0.5, easing=easing.quad)
        self.animate('opacity', 0, time=0.5)

    def appendPoint(self, point):
        self.points += point

    def draw(self):
        self.maze.set_color(*self.color + (self.opacity * 2, ))
        drawPolygon(self.points, GL_LINE_LOOP, linewidth=2)
        return Flash.draw(self)

class Label(AnimatedObject):
    def __init__(self, maze, coord, text, size=60, rotation=-90):
        AnimatedObject.__init__(self, maze.timer)
        self.maze = maze
        self.coord = (coord + (0, 0, 0))[:3]
        self.text = text
        self.size = size
        self.rotation = rotation
        self.opacity = 1

    def draw(self):
        glPushMatrix()
        glScalef(self.maze.cell_width, self.maze.cell_height, 1)
        glTranslatef(*self.coord)
        glScalef(1 / self.maze.cell_width, 1 / self.maze.cell_height, 1)
        glRotatef(-self.rotation, 0, 0, 1)
        try:
            drawLabel(
                    self.text,
                    pos=(0, 1),
                    font_size=round(self.size * 1.1),
                    center=True,
                    color=(1, 1, 1, self.opacity / 2),
                )
            drawLabel(
                    self.text,
                    pos=(0, 1),
                    font_size=round(self.size),
                    center=True,
                    color=(0, 0, 1, self.opacity),
                )
        except Exception:
            pass
        glPopMatrix()
        return self.opacity

class Maze(Game):
    numLaps = 3

    def __init__(self, log=None, replay=None, **kwargs):
        Game.__init__(self, **kwargs)
        self.scale = 1
        self.rotation = 0
        self.board_decorations = []
        np.random.seed(self.random.randint(0, 2**32))
        self.times = [0, 0]
        self.activeSide = 0
        self.lap = 0

    def start(self, width, height, cell_size=30):
        self.cell_size = cell_size
        self.window_width = width
        self.window_height = height
        self.startGame()
        return dict(
                cell_size=cell_size,
            )

    def startGame(self):
        self.matrix = -numpy.transpose(
                numpy.array(maze(
                        self.window_width // self.cell_size,
                        self.window_height // self.cell_size
                    ), dtype=numpy.int8)
            )
        self.width, self.height = self.matrix.shape
        self.cell_width = self.window_width / self.width
        self.cell_height = self.window_height / self.height
        self.cell_size = (self.cell_width + self.cell_height) / 2
        for x in range(self.width):
            self.matrix[x, 0] = self.matrix[x, self.height - 1] = -2
        for y in range(self.height):
            self.matrix[0, y] = self.matrix[self.width - 1, y] = -2
        self.matrix[0, 1] = -3
        self.matrix[self.width - 1, self.height - 2] = -4
        self.start_point = self.width - 2, self.height - 2
        self.start_cranny = self.width - 1, self.height - 2
        self.recompute_set = set()
        self.recompute_set.add(self.start_point)
        self.all_indices = []
        self.balls = []
        self.touches = {}
        self.decorations = []
        self.tries = set(), set()

        self.bdist = self.dist = self.matrix
        self.setWalls(self.matrix)

        self.home_size = 0

        self.timer.schedule(1, lambda: self.countdown(5))

        self.active = True

    @property
    def time(self):
        return self.times[self.activeSide]

    @time.setter
    def time(self, newValue):
        self.times[self.activeSide] = newValue

    def countdown(self, num):
        print num
        label = Label(self, (self.start_point[0] - 2, self.start_point[1] - 2), str(num) if num else 'Go!')
        label.animate('size', label.size * 10, time=1, dt=0.5, easing=easing.quart)
        label.animate('opacity', 0, time=1, dt=0.5, easing=easing.quad)
        self.decorations.append(label)
        if num > 0:
            self.timer.schedule(1, lambda: self.countdown(num - 1))
        else:
            self.animate('home_size', 4, time=1)

    def setWalls(self, matrix):
        """Set walls and solve the maze. No-op if maze is unsolvable
        """
        mstart = numpy.zeros(matrix.shape + (3,), dtype=numpy.int32)
        mstart[self.start_point + (0,)] = mstart[(1, 1, 1)] = 1
        mstart[self.start_cranny + (2,)] = 1
        mstart[self.start_point + (2,)] = 1
        for ball in self.balls:
            mstart[int(ball.coord[0]), int(ball.coord[1]), 2] = 1
        corridors = matrix >= 0
        m = solvemaze(corridors, mstart)
        if m is None:
            return False
        m[self.start_point + (0,)] = m[(1, 1, 1)] = 1
        self.bdist = m[:, :, 2]
        self.dist = m[:, :, 1]
        m = m[:, :, 0]
        self.matrix = numpy.select([matrix < 0, True], [matrix, m])
        self.solve_timer = 0
        return True

    def update(self, dt):
        for ball in self.balls:
            ball.update(dt)
        self.solve_timer += dt
        if self.solve_timer > 0.1:
            self.setWalls(self.matrix)
        for ball in self.balls:
            if ball.isOut():
                self.time += dt

    def setWall(self, coord, create):
        if coord == self.start_cranny or coord == (2, 1):
            return False
        m = self.matrix.copy()
        if create and self.matrix[coord] >= 0:
            m[coord] = -1
            return self.setWalls(m)
        elif not create and self.matrix[coord] == -1:
            m[coord] = 0
            return self.setWalls(m)
        else:
            return False

    darkWalls = True
    def swapColorScheme(self, darkWalls=None):
        if darkWalls is None:
            darkWalls = not self.darkWalls
        self.darkWalls = darkWalls
        self.log('darkWalls', darkWalls)
    swapColorScheme.name = 'Swap color scheme'
    replay_darkWalls = swapColorScheme

    def tileColor(self, x, y):
        m = 0
        if self.matrix[x, y] == -4:
            return self.tileColor(x - 1, y)
        elif self.matrix[x, y] == -3:
            return self.tileColor(x + 1, y)
        elif self.matrix[x, y] < 0:
            if self.darkWalls:
                return 0, 0, 0
            else:
                return 1, 1, 1
        else:
            m = self.matrix[x, y]
            d = self.dist[x, y]
            b = self.bdist[x, y]
            mmax = self.matrix.max()
            dmax = self.dist.max()
            bmax = self.bdist.max()
            r = m / mmax
            g = 1 - d / dmax
            b = 1 - min(m, b) / min(mmax, bmax)
            if self.darkWalls:
                return 1 - (numpy.array((max(g, b), max(r, b), max(r, g))) ** 8) / 2
            else:
                return numpy.array((r, g, b)) ** 8

    def draw(self):
        if self.scale <= 1:
            for i, time in enumerate(self.times):
                with gx_matrix:
                    glTranslatef((1 - i) * self.window_width, self.window_height / 2, 0)
                    glRotatef(i * 180 + 90, 0, 0, 1)
                    if self.activeSide == 0:
                        color = 1, 1, 1, 1
                        ltext = "Game %s/%s" % (self.lap + 1, self.numLaps)
                    else:
                        if time == max(self.times):
                            ltext = "Second place"
                            color = 0.7, 0.75, 0.75, 1
                        else:
                            ltext = "Winner!"
                            color = 0.94, 0.71, 0.098, 1
                    drawLabel(
                            formatTime(time),
                            pos=(0, 60),
                            font_size=50,
                            center=True,
                            color=color,
                        )
                    drawLabel(
                            ltext,
                            pos=(0, 130),
                            font_size=50,
                            center=True,
                            color=color,
                        )
        with gx_matrix:
            glTranslatef(self.window_width / 2, self.window_height / 2, 0)
            glScalef(self.scale, self.scale, 1)
            glRotatef(self.rotation, 0, 0, 1)
            glTranslatef(-self.window_width / 2, -self.window_height / 2, 0)

            self.set_color(1, 1, 1)
            for y in range(self.height):
                y_coord = y * self.cell_height
                for x in range(self.width):
                    x_coord = x * self.cell_width
                    self.set_color(*self.tileColor(x, y))
                    drawRectangle((x_coord, y_coord), (self.cell_width, self.cell_height))

            for ball in self.balls:
                ball.draw()

            if self.home_size:
                home_size = self.home_size * self.cell_size
                startCoord = (self.window_width, self.window_height)
                self.set_color(0, 0, 1, 0.1)
                drawCircle(startCoord, home_size)
                self.set_color(0, 0, 1, 0.3)
                drawCircle(startCoord, home_size, linewidth=2)

            self.decorations = [d for d in self.decorations if d.draw()]

        self.board_decorations = [d for d in self.board_decorations if d.draw()]

    def apply_transformation(self, x, y):
        if self.rotation % 360 == 0:
            return x, y
        else:
            return self.window_width - x, self.window_height - y

    def getTile(self, x, y):
        return x * self.width / self.window_width, y * self.height / self.window_height

    def touchDown(self, touch):
        if not self.active:
            return
        x, y = start = self.apply_transformation(touch.x, touch.y)
        print 'down', x, y
        tileCoord = self.getTile(x, y)
        for ball in self.balls:
            if ball.hittest(tileCoord):
                self.touches[touch.id] = dict(role='ball', ball=ball)
                ball.touched = True
                self.touchMove(touch)
                return
        if sum((numpy.array(tileCoord) - (self.width, self.height)) ** 2) < self.home_size ** 2:
            ball = Ball(self, self.start_cranny)
            self.balls.append(ball)
            self.touches[touch.id] = dict(role='ball', ball=ball)
            ball.touched = True
            self.touchMove(touch)
            return
        else:
            startCoord = numpy.floor(numpy.array(tileCoord))
            flash = BuildStartFlash(
                    self,
                    (startCoord + 0.5) * (self.cell_width, self.cell_height),
                    color=(1, 1, 0),
                    point=(x, y),
                )
            self.decorations.append(flash)
            self.touches[touch.id] = dict(
                    role='gesture',
                    points=[],
                    flash=flash,
                    startCoord=startCoord,
                )
            self.touchMove(touch)

    def touchMove(self, touch):
        if not self.active:
            return
        x, y = start = self.apply_transformation(touch.x, touch.y)
        print 'move', x, y
        tileCoord = self.getTile(x, y)
        try:
            d = self.touches[touch.id]
        except KeyError:
            return
        if d['role'] == 'wall':
            for ball in self.balls:
                if ball.blocks(numpy.array(tileCoord) + 0.5):
                    return
            key = int(tileCoord[0]), int(tileCoord[1])
            if key in self.tries[d['build']]:
                return
            try:
                build = self.matrix[tileCoord] > 0
            except IndexError:
                return
            else:
                if self.setWall(tileCoord, d['build']):
                    self.tries = set(), set()
                else:
                    self.tries[d['build']].add(key)
        elif d['role'] == 'gesture':
            pts = d['points']
            pts.append((x, y))
            d['flash'].appendPoint((x, y))
            length = sum(
                    numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    for (x1, y1), (x2, y2)
                    in zip(pts, pts[1:])
                )
            (x1, y1), (x2, y2) = pts[-1], pts[0]
            remainingLength = numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            perimeter = length + remainingLength
            signedArea = sum(
                    x1 * y2 - x2 * y1
                    for (x1, y1), (x2, y2)
                    in zip(pts, (pts[1:] + [pts[0]]))
                ) / 2
            areaInSquares = signedArea / self.cell_width / self.cell_height
            # Require a loop of area >= N squares; give an indication of
            # how far along it the drawer is
            N = 3
            l = (perimeter - remainingLength * 2)
            if l <= 0:
                indication = 0
            else:
                indication = areaInSquares * l / perimeter
                indication /= N
                if indication >= 1:
                    indication = 1
                if indication < -1:
                    indication = -1
            d['build'] = (indication < 0)
            if bool(indication > 0) ^ bool(self.darkWalls):
                d['flash'].color = (1, 1 - abs(indication), 0)
            else:
                d['flash'].color = (1 - abs(indication), 1, 0)
            if abs(areaInSquares) > N and (
                    numpy.floor(tileCoord) == d['startCoord']
                ).all():
                d['role'] = 'wall'
                d['flash'].explode()
                self.touchMove(touch)
        elif d['role'] == 'ball':
            ball = d['ball']
            ball.target = tileCoord

    def touchUp(self, touch):
        print 'up', touch.x, touch.y
        try:
            d = self.touches[touch.id]
            del self.touches[touch.id]
        except KeyError:
            return
        if d['role'] == 'ball':
            ball = d['ball']
            ball.target = ball.coord
            ball.touched = False
        try:
            d['flash'].end()
        except KeyError:
            pass

    def drawStateContents(self, w, r, *args, **kwargs):
        b = border = 4
        set_color(0, 0, 0, 1)
        complexity = self.matrix.max()
        #drawLabel("The maze", pos=(0, r / 2), font_size=10, center=True, color=(0, 0, 0))

        c_x, c_y = -(w - r) / 2, b
        c_w, c_h = (w - r), r - 2 * b

        if self.balls:
            completeness = 1 - ((self.bdist[1, 1] - 1) / self.dist[self.start_point])
            if completeness > 1:
                pass
            elif completeness >= 0:
                set_color(0, 0, 0.75, .5)
                drawRectangle(pos=(c_x, c_y), size=(c_w * min(1, completeness), c_h))
            else:
                set_color(0.75, 0, 0, .5)
                if completeness < -1:
                    drawRectangle(pos=(c_x, c_y), size=(c_w, c_h))
                    completeness += 1
                    completeness /= 2
                    set_color(0, 0, 0)
                w = c_w * max(0, abs(completeness))
                drawRectangle(pos=(c_x + c_w - w + 1, c_y), size=(w, c_h))
            set_color(0, 0, 0, 1)
            drawRectangle(pos=(c_x, c_y), size=(c_w + 1, c_h), style=GL_LINE_LOOP)
        else:
            #drawLabel("Start at blue corner!", pos=(0, r / 2), font_size=10, center=True, color=(0, 0, 0))
            pass
        drawLabel(formatTime(self.time), pos=(0, r / 2), font_size=10, center=True, color=(0, 0, 0))

    def isWall(self, tileCoord):
        try:
            return -2 <= self.matrix[int(tileCoord[0]), int(tileCoord[1])] < 0
        except IndexError:
            return True

    def win(self):
        self.active = False
        self.animate('home_size', 0, time=1)
        self.animate('rotation', 180, time=2, additive=True, easing=easing.sin.outIn)
        self.animate('scale', 0.2, time=1, easing=easing.sin.out)
        if self.lap + 1 < self.numLaps or self.activeSide == 0:
            self.animate('scale', 1, time=1, dt=1, easing=easing.sin)
            self.timer.schedule(1, self.startGame)

            def swapSide():
                if self.activeSide == 0:
                    self.activeSide = 1
                else:
                    self.activeSide = 0
                    self.lap += 1
            self.timer.schedule(2, swapSide)
        else:
            self.animate('scale', 0, time=1, dt=1, easing=easing.sin.out)

        for i in range(30):
            if self.rotation % 360:
                start = self.window_width, self.window_height
                direction = -1
            else:
                start = 0, 0
                direction = 1
            flash = Flash(self, start, color=(random.random() ** 2, random.random() ** 2, random.random() ** 0.5))
            flash.radius = (random.random() + 1) * self.cell_size / 2
            lifetime = 0.5 + random.random()
            r = direction * (self.window_height + self.window_width) / 4
            flash.animate('coord', (random.random() * r, random.random() * r), time=lifetime, easing=easing.quad.out, additive=True)
            flash.animate('opacity', 0, time=lifetime)
            self.board_decorations.append(flash)

    def menuContents(self, paused):
        return [self.swapColorScheme]

def formatTime(time):
    return "{0:.0f}:{1:=04.1f}".format(*divmod(time, 60))

registerPyMTPlugin(Maze, globals())

if __name__ == '__main__':
    main()
