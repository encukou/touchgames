#! /usr/bin/env python

from __future__ import division


import numpy as np
from numpy.random import random_integers as rnd
import matplotlib.pyplot as plt
 
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
from OpenGL.GL import glColor3f, glColor4f, GL_LINE_LOOP

import touchgames
from touchgames.game import Game
from touchgames.gamecontroller import registerPyMTPlugin

class Ball(object):
    radius = 0.4

    def __init__(self, maze, coord):
        self.maze = maze
        self.coord = numpy.array(coord) + 0.5
        self.target = self.coord

    def draw(self):
        drawCoord = (self.coord) * (
                        self.maze.cell_width,
                        self.maze.cell_height,
                    )
        drawRadius = self.radius * self.maze.cell_size
        self.maze.set_color(0, 0, 1)
        drawCircle(drawCoord, drawRadius)
        self.maze.set_color(0, 0, 0, 0.1)
        drawCircle(drawCoord, drawRadius * 5, linewidth=3)

    def hittest(self, tileCoord):
        return sum((tileCoord - self.coord) ** 2) <= (
                (self.radius * 5) ** 2
            )

    def update(self, dt):
        distance_covered = 0
        while distance_covered < dt * 5:
            if abs(sum(self.target - self.coord)) < 0.1:
                break
            distance_covered += self.dragBy(self.target - self.coord)

    def dragBy(self, delta):
        # returns distance covered
        length = numpy.sqrt(sum(delta ** 2))
        max_len = self.radius
        if length > max_len ** 2:
            delta = delta / length * max_len
        radius = self.radius
        for i in (0, 1):
            # Check the up/down & left/right points of the circle
            pt = [0, 0]
            pt[i] = radius
            if self.maze.isWall(self.coord + delta + pt):
                delta[i] = int(self.coord[i] + delta[i] + 1) - radius - self.coord[i]
            if self.maze.isWall(self.coord + delta - pt):
                delta[i] = int(self.coord[i] + delta[i]) + radius - self.coord[i]
        sign = numpy.sign(delta)
        self.coord += delta
        return length

class Maze(Game):
    def start(self, width, height):
        cell_size = 20
        self.window_width = width
        self.window_height = height
        self.matrix = -numpy.transpose(
                numpy.array(maze(width // cell_size, height // cell_size), dtype=numpy.int8)
            )
        self.width, self.height = self.matrix.shape
        self.cell_width = width / self.width
        self.cell_height = height / self.height
        self.cell_size = (self.cell_width + self.cell_height) / 2
        for x in range(self.width):
            self.matrix[x, 0] = self.matrix[x, self.height - 1] = -2
        for y in range(self.height):
            self.matrix[0, y] = self.matrix[self.width - 1, y] = -2
        self.matrix[0, 1] = -3
        self.matrix[self.width - 1, self.height - 2] = -4
        self.start_point = self.width - 2, self.height - 2
        self.recompute_set = set()#HintQueue(self)
        self.recompute_set.add(self.start_point)
        self.all_indices = []
        self.setWalls(self.matrix)
        self.touches = {}
        self.balls = [Ball(self, self.start_point)]

    def setWalls(self, matrix):
        """Set walls and solve the maze. No-op if maze is unsolvable
        """
        infinity = 2**30
        m = numpy.zeros(matrix.shape + (3,), dtype=numpy.int32) + infinity
        mat = numpy.dstack((matrix, matrix, matrix))
        corridors = mat >= 0
        print '{'
        for i in range(10 * (self.width + self.height)):
            m[self.start_point + (0,)] = m[(1, 1, 1)] = 1
            m = numpy.select([numpy.logical_and(corridors, m < infinity)], [m], infinity)
            m = numpy.minimum(
                    numpy.minimum(numpy.roll(m, 1, 0), numpy.roll(m, -1, 0)),
                    numpy.minimum(numpy.roll(m, 1, 1), numpy.roll(m, -1, 1)),
                ) + 1
            m = numpy.select([corridors], [m], 0)
            if m[:, :, :2].max() < infinity:
                break
        else:
            print '} maxed out!'
            return False
        m[self.start_point + (0,)] = m[(1, 1, 1)] = 1
        self.bdist = m[:, :, 2]
        self.dist = m[:, :, 1]
        m = m[:, :, 0]
        self.matrix = numpy.select([matrix < 0, m < infinity], [matrix, m], 0)
        print '}'
        return True

    def update(self, dt):
        for ball in self.balls:
            ball.update(dt)

    def setWall(self, coord, create):
        m = self.matrix.copy()
        if create and self.matrix[coord] >= 0:
            m[coord] = -1
            return self.setWalls(m)
        elif not create and self.matrix[coord] == -1:
            m[coord] = 0
            return self.setWalls(m)
        else:
            return False

    def draw(self):
        self.set_color(1, 1, 1)
        for x in range(self.width):
            x_coord = x * self.cell_width
            #pymt.drawLine((x_coord, 0, x_coord + .1, self.window_height))
        for y in range(self.height):
            self.set_color(1, 1, 1)
            y_coord = y * self.cell_height
            #pymt.drawLine((0, y_coord - .1, self.window_width, y_coord + .1))
            for x in range(self.width):
                x_coord = x * self.cell_width
                m = 0
                if self.matrix[x, y] == -4:
                    self.set_color(1, 0, 0)
                elif self.matrix[x, y] == -3:
                    self.set_color(0, 1, 0)
                elif self.matrix[x, y] < 0:
                    m = 1
                    self.set_color(1, 1, 1)
                else:
                    m = self.matrix[x, y] / self.matrix.max()
                    d = self.dist[x, y] / self.dist.max()
                    b = self.bdist[x, y] / self.bdist.max()
                    self.set_color(1 - m, 1 - d, 1 - b)
                drawRectangle((x_coord, y_coord), (self.cell_width, self.cell_height))
                self.set_color(1, 1, 1)
        for ball in self.balls:
            ball.draw()

    def getTile(self, x, y):
        return x * self.width / self.window_width, y * self.height / self.window_height

    def touchDown(self, touch):
        x, y = start = touch.x, touch.y
        tileCoord = self.getTile(x, y)
        for ball in self.balls:
            if ball.hittest(tileCoord):
                self.touches[touch.id] = dict(
                        role='ball',
                        ball=ball,
                        initial=tileCoord - ball.coord,
                    )
                return
        try:
            build = self.matrix[tileCoord] > 0
        except IndexError:
            return
        else:
            self.setWall(tileCoord, build)
        self.touches[touch.id] = dict(role='wall', build=build)

    def touchMove(self, touch):
        x, y = start = touch.x, touch.y
        tileCoord = self.getTile(x, y)
        try:
            d = self.touches[touch.id]
        except KeyError:
            return
        if d['role'] == 'wall':
            try:
                build = self.matrix[tileCoord] > 0
            except IndexError:
                return
            else:
                self.setWall(tileCoord, d['build'])
        if d['role'] == 'ball':
            ball = d['ball']
            ball.target = tileCoord - d['initial']

    def touchUp(self, touch):
        try:
            d = self.touches[touch.id]
            del self.touches[touch.id]
        except KeyError:
            return
        if d['role'] == 'ball':
            ball = d['ball']
            ball.target = ball.coord

    def set_color(self, *rgb):
        if len(rgb) == 3:
            glColor3f(*self.color(*rgb))
        else:
            glColor4f(*self.color(*rgb))

    def drawStateContents(self, w, r, *args, **kwargs):
        b = border = 4
        set_color(0, 0, 0, 1)
        complexity = self.matrix.max()
        drawLabel("The maze", pos=(0, r / 2), font_size=10, center=True, color=(0, 0, 0))

        c_x, c_y = -(w - r) / 2, b
        c_w, c_h = (w - r), r - 2 * b

        #set_color(0, 0.75, 0.25, .75)
        #drawRectangle(pos=(c_x, c_y), size=(c_w * min(1, self.completeness), c_h))
        #set_color(0, 0, 0, 1)
        #drawRectangle(pos=(c_x, c_y), size=(c_w + 1, c_h), style=GL_LINE_LOOP)

    def isWall(self, tileCoord):
        try:
            return self.matrix[int(tileCoord[0]), int(tileCoord[1])] < 0
        except IndexError:
            return True

registerPyMTPlugin(Maze, globals())

if __name__ == '__main__':
    main()
