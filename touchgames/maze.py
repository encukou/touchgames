#! /usr/bin/env python
# Encoding: UTF-8

from __future__ import division

import random
import itertools
import math
from collections import defaultdict
import sys

import numpy
from numpy.random import random_integers

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import (Color, Ellipse, Line, Rectangle, Point,
    Rotate, Translate, Scale, PushMatrix, PopMatrix)
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.graphics.transformation import Matrix
from kivy.graphics.instructions import Canvas

from touchgames.mazesolver import solvemaze
from touchgames.util import FilledCircle, HollowCircle
from touchgames.replay import Logger

COUNTDOWN_START = 5
MAZE_CELL_SIZE = 30
BALL_SPEED = 10  # tiles per second
BALL_TOUCH_RADIUS = 1.5  # tiles
BALL_ZOC_RADIUS = 7  # tiles
MIN_LOOP_AREA = 3  # tiles squared
NUM_ROUNDS = 2

def yield_groups(source, n):
    """Divide an iterator into tuples of n consecutive elements

    For example, with n=2: 1, 2, 3, 4, 5, 6 -> (1, 2), (3, 4), (5, 6)
    """
    return itertools.izip(*[iter(source)] * n)

class BallSource(Widget):
    """The quarter-circle in the corner that balls are spawned from

    Will be initially zero-sized, then grown when `grow` is called.
    Provides collision detection, which is checked when balls are generated.
    """
    def __init__(self, **kwargs):
        super(BallSource, self).__init__(**kwargs)

    def grow(self, size):
        """Animate the widget to `size` over the time of 1 second
        """
        animation = Animation(size=(size, size), t='in_cubic', duration=1)
        animation.start(self)
        tick = schedule_tick(self.tick)
        Clock.schedule_once(lambda dt: Clock.unschedule(tick), 2)

    def tick(self, dt):
        """Redraw the widget (kivy won't update it automatically)
        """
        size = self.size[0]
        self.canvas.clear()
        with self.canvas:
            Color(0, 0, 1, 0.5)
            Ellipse(
                    pos=[p - s for p, s in zip(self.pos, self.size)],
                    size=[x * 2 - 1 for x in self.size],
                )
            Color(1, 1, 1, 1)
            HollowCircle(self.pos, self.size[0])

    def collide_point(self, x, y):
        px, py = self.pos
        sx, sy = self.size
        return (px - x) ** 2 + (py - y) ** 2 < sx ** 2

class TickingWidget(Widget):
    """A widget that calls its tick() method on each frame.

    Be sure to remove it from the widget hierarchy when it's done.
    """
    def __init__(self, **kwargs):
        super(TickingWidget, self).__init__(**kwargs)
        self._tick = None
        self.bind(parent=self.on_parent_changed)

    def on_parent_changed(self, instance, value):
        if self._tick:
            self._tick.unschedule()
            self._tick = None
        if value:
            self._tick = schedule_tick(self.tick)

    def tick(self, dt):
        """The base class implementation redraws the widget's canvas.
        """
        self.canvas.ask_update()

class Ball(TickingWidget):
    """The ball that the solver moves through the maze.

    The ball has two collision-detection methods: the standard collide_point()
    for the area where the ball accepts touches, and collide_zoc() for the
    “zone of control”. The zone of coltrol grows when the ball is touched, and
    shrinks when the touch is released.

    When touched, the ball will move towards the touch at BALL_SPEED tiles per
    second (but maze walls will block it).
    """
    def __init__(self, parent, **kwargs):
        super(Ball, self).__init__(**kwargs)
        self.touch_uid = None
        self.target_pos = self.pos

        radius = self.radius = parent.cell_size * 0.3
        self.handle_radius = parent.cell_size * BALL_TOUCH_RADIUS
        self.zoc_radius = self.radius
        with self.canvas:
            Color(0, 0, 1, 0.5)
            Ellipse(pos=(-radius, -radius), size=(radius * 2, radius * 2))
            Color(0, 0, 0, 1)
            HollowCircle((0, 0), radius, 18)
            Color(0.5, 0.5, 0.5, 0.4)
            FilledCircle((0, 0), self.handle_radius)
            HollowCircle((0, 0), self.handle_radius, 18)
            self.scale_instruction = Scale(self.zoc_radius)
            Color(1, 1, 1, 0.2)
            FilledCircle((0, 0), 1)
            Color(0.5, 0.5, 0.5, 0.4)
            HollowCircle((0, 0), 1, 32)
        with self.canvas.before:
            PushMatrix()
            self.translation_instruction = Translate(0, 0, 0)
        with self.canvas.after:
            PopMatrix()

    def collide_point(self, x, y):
        px, py = self.pos
        return (px - x) ** 2 + (py - y) ** 2 < self.handle_radius ** 2

    def collide_zoc(self, x, y):
        px, py = self.pos
        return (px - x) ** 2 + (py - y) ** 2 < self.zoc_radius ** 2

    def tick(self, dt):
        if not self.parent:
            return
        # Try to cover the required distance. But if it's not done in 50
        # iterations, give up.
        remaining_distance = dt * BALL_SPEED * self.parent.cell_size
        for i in range(50):
            if remaining_distance <= 0.01 or self.pos == self.target_pos:
                break
            distance_covered = self.move_step(remaining_distance)
            if distance_covered == 0:
                break
            remaining_distance -= distance_covered
        if self.translation_instruction.xy != self.pos:
            # Update the canvas if the ball moved
            self.translation_instruction.xy = self.pos
            self.canvas.ask_update()
        if self.scale_instruction.scale != self.zoc_radius:
            # Update the canvas if the ZOC was resized
            self.scale_instruction.scale = self.zoc_radius
            self.canvas.ask_update()
        if not self.parent.ball_source.collide_point(*self.pos):
            # If the ball is outside the initial area, add time to the player's
            # clock
            self.parent.add_time(dt)
        if self.x < self.parent.cell_size:
            # IF the ball is in the goal area, the round ends
            self.parent.win()

    def move_step(self, remaining_distance):
        """Move a little towards the touch position, return distance covered

        `remaining_distance` is the maximum distance to move

        This implements one iteration of the moving, and is called enough times
        for the sum of the returned values is big enough.

        Both remaining_distance and the return value are in pixels, not tiles.
        """
        radius = self.radius
        pos = numpy.array(self.pos)
        # The distance we want to cover
        delta = self.target_pos - pos
        distance = numpy.sqrt(sum(delta ** 2))
        # Only move a little bit each time, so walls are checked correctly
        max_distance = min(remaining_distance, radius / 2)
        if distance > max_distance:
            delta = delta / distance * max_distance
            distance = max_distance
        pos += delta
        # From now, we will deal with tile coordinates instead of pixels
        tile_coord = numpy.array(self.parent.pixel_to_tile(pos))
        tile_radius = self.parent.pixel_to_tile((radius, radius))
        # Check the upper/lower & left/right points of the circle
        # if one of them is in a wall, "snap" the ball to that wall
        for axis in (0, 1):
            pt = [0, 0]
            pt[axis] = tile_radius[axis]
            if self.parent.wall_at_tile(tile_coord + pt):
                tile_coord[axis] = int(tile_coord[axis]) + 1 - pt[axis]
            if self.parent.wall_at_tile(tile_coord - pt):
                tile_coord[axis] = int(tile_coord[axis]) + pt[axis]
        # Get the closest grid intersection
        corner = numpy.array(tile_coord).round()
        # Get a point in the tile "behind" this clocest corner
        tile_behind_corner = 2 * corner - tile_coord
        # Check if there is a wall on that tile
        if self.parent.wall_at_tile(tile_behind_corner):
            vector_to_corner = corner - tile_coord
            # If part of the ball is inside a corner wall, push it back
            # XXX: This doesn't take into account that the ball can be slightly
            # elliptical in tile coordinates. (This isn't likely to matter,
            # though.)
            avg_radius = sum(tile_radius) / 2
            if sum(vector_to_corner ** 2) < avg_radius ** 2:
                distance_to_corner = numpy.sqrt(sum(vector_to_corner ** 2))
                direction_to_corner = vector_to_corner / distance_to_corner
                pushback_distance = avg_radius - distance_to_corner
                tile_coord -= direction_to_corner * pushback_distance
                pushed_back = True
        # Convert back to pixel coordinates
        self.pos = self.parent.tile_to_pixel(tile_coord)
        return distance

    def on_touch_down(self, touch, force=False):
        """Called for a new touch

        `force`: don't check that the touch is near the ball; assume it is.
        """
        if force or self.collide_point(touch.x, touch.y):
            if self.touch_uid:
                self.touch_uid = None
            self.touch_uid = touch.uid
            self.on_touch_move(touch)
            zoc_radius = self.parent.cell_size * BALL_ZOC_RADIUS
            animation = Animation(zoc_radius=zoc_radius, duration=0.5)
            animation.start(self)
            return True

    def on_touch_move(self, touch):
        if touch.uid == self.touch_uid:
            self.target_pos = touch.pos
            return True

    def on_touch_up(self, touch):
        if touch.uid == self.touch_uid:
            self.touch_uid = None
            self.target_pos = self.pos
            animation = Animation(zoc_radius=self.handle_radius,
                    t='in_cubic', duration=1)
            animation.start(self)
            return True

class Builder(Widget):
    """The “line” for modifying the maze

    Follows a touch, and either builds or destroys corridors (based on the
    `build` argument). Disappears on touch up.

    The line is either green (for building) or red (for destroying).
    If the line goes through a ball zone of control, it is inactive (no
    building/destroying). This is indicated graphically by turning the line
    gray. The inactive part stays gray even if the ball is moved.

    The line is kept at a maximum length, by fading out the “tail”.
    """
    def __init__(self, build, **kwargs):
        super(Builder, self).__init__(**kwargs)
        # points: a list of (x, y, active) triples
        self.points = [(self.pos[0], self.pos[1], True)]
        # build: True for building, False for destroying
        self.build = build
        # opacity: animated to 0 for a fade-out
        self.opacity = 1

    def on_touch_down(self, touch, force=False):
        if force:
            self.max_points = int(self.parent.cell_size / 3 * 10)
            self.touch_uid = touch.uid
        if self.touch_uid == touch.uid:
            #touch.grab(self)
            return True

    def on_touch_move(self, touch):
        if self.touch_uid == touch.uid:
            old_x, old_y, old_active = self.points[-1]
            parent = self.parent
            for x, y in calculate_points(old_x, old_y, touch.x, touch.y,
                    spacing=3):
                ok = True
                for ball in self.parent.balls:
                    if ball.collide_zoc(x, y):
                        ok = False
                        break
                self.points.append((x, y, ok))
                if ok:
                    tile_coord = parent.pixel_to_tile((x, y))
                    int_tile = int(tile_coord[0]), int(tile_coord[1])
                    # Try building/destroying
                    if parent.set_wall(int_tile, self.build):
                        # Successful change to the maze, show some SFX
                        particle_shower(
                                parent=self.parent,
                                type='build' if self.build else 'destroy',
                                pos=self.parent.tile_to_pixel((
                                        int_tile[0] + 0.5,
                                        int_tile[1] + 0.5,
                                    )),
                            )
            # Trim the line to the max length
            max_points = self.max_points
            self.points = self.points[-max_points:]
            self.redraw()
            return True

    def on_touch_up(self, touch):
        """Destroy the widget when the touch disappears.

        Includes a fade-out animation.
        """
        if self.touch_uid == touch.uid:
            Animation(opacity=0, duration=0.25).start(self)
            tick = schedule_tick(self.redraw)
            def die(dt):
                Clock.unschedule(tick)
                if self.parent:
                    self.parent.remove_widget(self)
            Clock.schedule_once(die, 0.25)
            return True

    def redraw(self, dt=None):
        """Redraw the whole line
        """
        self.canvas.clear()
        if self.build:
            r, g = 0, 1
        else:
            r, g = 1, 0
        with self.canvas:
            for opacity, (x, y, active) in zip(
                    range(self.max_points), reversed(self.points)):
                alpha = (1 - opacity / self.max_points) * self.opacity
                if active:
                    Color(r, g, 0, alpha)
                else:
                    Color(0.4 + r * 0.1, 0.4 + g * 0.1, 0.5, alpha)
                Point(points=(x, y), pointsize=5, source='particle.png')

class BuildLoop(TickingWidget):
    """A widget that allows a player to draw a loop to begin modifying the maze

    Represented by a ring of dots spinning around the starting point, and a
    line traced by the touch. The color of these changes from yellow to green
    or red depending on the direction of the loop being drawn.

    Once the touch returns to the starting position, and the loop is
    sufficiently big, a corresponding Builder is created at that position.

    The game needs to know the orientation and area of the loop, and the player
    can draw arbitrary polygons instead of plain loops. A simple formula for
    the signed area of a polygon[1] is used, which works very well for both
    “proper” loops and any convex or self-intersecting drawings the player
    might draw.

    [1] http://mathworld.wolfram.com/PolygonArea.html
    """
    def __init__(self, spin, **kwargs):
        super(BuildLoop, self).__init__(**kwargs)
        self.spin = spin
        self.touch_uid = None

    def on_touch_down(self, touch, force=False):
        if force:
            self.touch_uid = touch.uid
        if self.touch_uid == touch.uid:
            #touch.grab(self)
            with self.canvas.before:
                PushMatrix()
            with self.canvas.after:
                PopMatrix()
            with self.canvas:
                # Create the ring of “satellite” points
                self.color_instruction = Color(1, 1, 0, 0)
                self.points = Point(points=self.pos, pointsize=5,
                    source='particle.png')
                Translate(self.pos[0], self.pos[1], 0)
                self.scale_instruction = Scale(self.parent.cell_size * 10)
                self.rotate_instruction = Rotate(0, 0, 0, 1)
                self.rscale_instruction = Scale(1)
                num_satellites = 9
                satellites = []
                for i in range(num_satellites):
                    angle = i / num_satellites * 2 * math.pi
                    satellites.extend([
                            math.cos(angle) * 1.3,
                            math.sin(angle) * 1.3,
                        ])
                Point(points=satellites, pointsize=0.2, source='particle2.png')

            # A scale-in animation
            animation = Animation(scale=self.parent.cell_size / 3 * 2,
                    t='out_cubic', duration=0.2)
            animation.start(self.scale_instruction)

            # A fade-in animation
            animation = Animation(a=0.8, t='out_cubic', duration=0.2)
            animation.start(self.color_instruction)

            # A practically infinite spinning animation
            animation = Animation(angle=200000 * self.spin, duration=1000)
            animation.start(self.rotate_instruction)
            return True

    def on_touch_move(self, touch):
        if self.touch_uid == touch.uid:
            points = self.points.points
            oldx, oldy = points[-2], points[-1]
            points = calculate_points(oldx, oldy, touch.x, touch.y, spacing=3)
            if points:
                add_point = self.points.add_point
                for x, y in points:
                    add_point(x, y)

            # Calculate the area of the polygon being drawn (closing it with
            # a line from end to beginning); the sign of the result represents
            # the orientation.
            signed_area = sum(
                    x1 * y2 - x2 * y1
                    for (x1, y1), (x2, y2)
                    in zip(
                            yield_groups(self.points.points, 2),
                            yield_groups(self.points.points[2:] +
                                self.points.points[:2], 2),
                        )
                ) / 2
            # Compute the area in tiles²
            tile_square_unit = self.parent.cell_width * self.parent.cell_height
            area_in_tiles = signed_area / tile_square_unit
            # `done`: the percentage to which the loop is done; signed as above
            done = area_in_tiles / MIN_LOOP_AREA
            if done < 0:
                # Clockwise loop; green
                if done < -1:
                    done = -1
                color = 1 + done, 1, 0
            else:
                # Counter-clockwise loop; red
                if done > 1:
                    done = 1
                color = 1, 1 - done, 0
            # Set the color
            self.color_instruction.rgb = color
            # Check if we're back at the starting tile with a big enough loop
            if abs(done) == 1:
                x, y = self.parent.pixel_to_tile(self.pos)
                end_x, end_y = self.parent.pixel_to_tile(touch.pos)
                if int(x) == int(end_x) and int(y) == int(end_y):
                    # Create a builder
                    touch.ungrab(self)
                    builder = Builder(pos=self.pos, build=done < 0)
                    self.parent.add_widget(builder)
                    builder.on_touch_down(touch, force=True)
                    builder.on_touch_move(touch)
                    self.die()
            return True

    def on_touch_up(self, touch):
        if self.touch_uid == touch.uid:
            touch.ungrab(self)
            self.die()
            return True

    def die(self):
        """Ending animation and destruction of the widget
        """
        # Scale-out animation
        animation = Animation(scale=self.parent.cell_size * 15,
                t='in_cubic', duration=0.15)
        animation.start(self.scale_instruction)

        # Fade-out animation
        animation = Animation(a=0, duration=0.15)
        animation.start(self.color_instruction)

        # Disown the widget
        def _die(dt):
            if self.parent:
                self.parent.remove_widget(self)
        Clock.schedule_once(_die, 0.15)

class MazeBoard(TickingWidget):
    """A maze for a single turn in the game

    MazeBoard contains the maze. Balls and builders are child widgets of a
    MazeBoard.
    """
    darkWalls = True

    def __init__(self):
        self.initialized = False
        super(MazeBoard, self).__init__()
        Clock.schedule_once(lambda dt: self.initialize())
        self.must_redraw = False
        self.time = 0

    def initialize(self):
        self.initialized = True

        try:
            self.window_width
            self.window_height
        except AttributeError:
            win = self.get_parent_window()

            # `window_width` & `window_height`: dimensions of the window, in pixels
            self.window_width = win.width
            self.window_height = win.height

        # `matrix`: the matrix holding the maze. Values in this matrix are:
        # - nonnegative numbers: corridors; the value represents the shortest
        #   distance to the exit (these will be set using set_walls later)
        # -1: regular (breakable) wall
        # -2: perimeter (unbreakable) wall
        # -3: exit
        # -4: entrance
        self.matrix = -numpy.transpose(
                numpy.array(create_maze(
                        self.window_width // MAZE_CELL_SIZE,
                        self.window_height // MAZE_CELL_SIZE,
                    ), dtype=numpy.int8)
            )
        # `width`/`height`: size of the maze, in tiles
        self.width, self.height = self.matrix.shape
        # `cell_width`/`cell_height`: size of a tile, in pixels
        self.cell_width = self.window_width / self.width
        self.cell_height = self.window_height / self.height
        # `cell_size`: average size of a tile, as a scalar
        self.cell_size = (self.cell_width + self.cell_height) / 2

        # Initialize perimeter walls and entrance/exit
        for x in range(self.width):
            self.matrix[x, 0] = self.matrix[x, self.height - 1] = -2
        for y in range(self.height):
            self.matrix[0, y] = self.matrix[self.width - 1, y] = -2
        self.matrix[0, 1] = -3
        self.matrix[self.width - 1, self.height - 2] = -4

        # `balls`: list of Ball widgets on this board
        self.balls = []

        # `start_point`, `start_cranny`: Coordinates of the entrance and the
        # tile next to it. (The corresponding coordinates for the exit are
        # simply (0, 1) and (1, 1))
        self.start_point = self.width - 2, self.height - 2
        self.start_cranny = self.width - 1, self.height - 2

        # `bdist`: same as `matrix` except positive numbers indicate shortest
        #  distance to the entrance or nearest ball
        # `dist`: ditto with shortest distance to just the entrance
        self.bdist = self.dist = self.matrix
        self.set_walls(self.matrix)

        # Initialize the graphic representation
        self.background_canvas = Canvas()
        self.canvas.add(self.background_canvas)
        self.draw()

        # Add the ball source (zero-sized initially)
        self.ball_source = BallSource(size=(0, 0),
                pos=(self.window_width, self.window_height))
        self.add_widget(self.ball_source)

        # Initialize the countdown for the solver
        Clock.schedule_once(lambda dt: self.countdown(COUNTDOWN_START), 1)

        # Redraw every 30th of a second
        Clock.schedule_interval(self.redraw, 0.03)

    def countdown(self, num):
        """The countdown for the solving player

        For positive `num`, shows the number near the maze entrance, animates
        it, and schedules a call to countdown(num-1) for one second.
        For num == 0, display 'Go!', animate it, and grow the ball source.
        """
        label = Label(
                text=str(num) if num else 'Go!',
                font_size=24,
                color=(0, 0, 1, 1),
            )
        with label.canvas.before:
            PushMatrix()
            Translate(
                    self.window_width - self.cell_size * 1,
                    self.window_height - self.cell_size * 5,
                    0,
                )
            Rotate(90, 0, 0, 1)
        with label.canvas.after:
            PopMatrix()
        animation = Animation(font_size=256, color=(0, 0, 1, 0),
                t='in_cubic', duration=1.5)
        animation.start(label)
        self.add_widget(label)
        if num > 0:
            Clock.schedule_once(lambda dt: self.countdown(num - 1), 1)
        else:
            self.ball_source.grow(self.cell_size * 3)
            if self.parent:
                self.parent.set_message(True,
                        u'Take a ball from the blue corner.')

    def add_time(self, time):
        """Add time to the solving player's clock
        """
        self.time += time
        if self.time > 15:
            # If the player is taking a long time, display a (hopefully
            # helpful) hint
            self.parent.set_message(True,
                    u"Take another ball if you're really stuck.")
        self.parent.add_time(time)

    def draw(self):
        """Draw the board.

        The colors are filled in later, in tick()
        """
        self.colors = {}
        self.background_canvas.clear()
        with self.background_canvas:
            for y in range(self.height):
                y_coord = y * self.cell_height
                for x in range(self.width):
                    x_coord = x * self.cell_width
                    self.colors[x, y] = Color(0, 0, 0)
                    Rectangle(pos=(x_coord, y_coord),
                            size=(self.cell_width, self.cell_height))
        self.redraw()

    def redraw(self, dt=None):
        """Called any time the maze's colors change
        """
        self.must_redraw = True

    def tick(self, dt=None):
        """Set all tile colors to the appropriate values
        """
        if not self.must_redraw:
            return
        else:
            self.must_redraw = False
        sys.stdout.flush()
        self.set_walls(self.matrix)
        for y in range(self.height):
            y_coord = y * self.cell_height
            for x in range(self.width):
                self.colors[x, y].rgba = self.tile_color(x, y)
        self.background_canvas.ask_update()

    def tile_color(self, x, y):
        """Get the color for the tile at (x, y)
        """
        if self.matrix[x, y] == -4:
            # Entrance; same color as neighboring tile
            return self.tile_color(x - 1, y)
        elif self.matrix[x, y] == -3:
            # Exit; same color as neighboring tile
            return self.tile_color(x + 1, y)
        elif self.matrix[x, y] < 0:
            # Wall, transparent black
            if self.darkWalls:
                return 0, 0, 0, 0
            else:
                # (with light walls; white)
                return 1, 1, 1, 1
        else:
            # Corridor; a light color based on distances to the entrance, exit
            # and nearest ball
            m = self.matrix[x, y]
            d = self.dist[x, y]
            b = self.bdist[x, y]
            mmax = self.matrix.max() or 1
            dmax = self.dist.max() or 1
            bmax = self.bdist.max() or 1
            r = m / mmax
            g = 1 - d / dmax
            b = 1 - min(m, b) / min(mmax, bmax)
            if self.darkWalls:
                a = (numpy.array((max(g, b), max(r, b), max(r, g), 0)) ** 8)
                return 1 - a / 2
            else:
                # (with light walls; a dark color)
                return numpy.array((r, g, b, 1)) ** 8

    def set_walls(self, matrix):
        """Set walls and solve the maze. No-op if maze is unsolvable.

        `matrix`: A 2D matrix with negative values for walls
        """
        mstart = numpy.zeros(matrix.shape + (3,), dtype=numpy.int32)
        mstart[self.start_point + (0,)] = mstart[(1, 1, 1)] = 1
        mstart[self.start_cranny + (2,)] = 1
        mstart[self.start_point + (2,)] = 1
        for ball in self.balls:
            tile_pos = self.pixel_to_tile(ball.pos)
            mstart[int(tile_pos[0]), int(tile_pos[1]), 2] = 1
        corridors = matrix >= 0
        m = solvemaze(corridors, mstart)
        if m is None:
            return False
        m[self.start_point + (0,)] = m[(1, 1, 1)] = 1
        self.bdist = m[:, :, 2]
        self.dist = m[:, :, 1]
        m = m[:, :, 0]
        self.matrix = numpy.select([matrix < 0, True], [matrix, m])

        # Clear cache of unsuccessful building attempts
        self.build_tries = defaultdict(set)
        return True

    def set_wall(self, coord, create):
        """Set (create== True) or destroy (create == False) a wall at coord

        If the operation would create an unsolvable maze, nothing is done.
        These unsuccessful attempts are cached in `build_tries` until the maze
        is changed
        """
        if coord in self.build_tries[create]:
            return False
        self.build_tries[create].add(coord)
        if coord == self.start_cranny or coord == (2, 1):
            return False
        try:
            current_state = self.matrix[coord]
        except IndexError:
            return False
        m = self.matrix.copy()
        if not create and current_state >= 0:
            # Build a wall
            m[coord] = -1
            rv = self.set_walls(m)
        elif create and current_state == -1:
            # Build a corridor
            m[coord] = 0
            rv = self.set_walls(m)
        else:
            return False
        if rv:
            self.redraw()
        return rv

    def pixel_to_tile(self, pos):
        """Convert pixel coordinates to tile coordinates

        Returns a pair of floats
        """
        return (
                pos[0] / self.cell_width,
                pos[1] / self.cell_height,
            )

    def tile_to_pixel(self, tile_coord):
        """Convert tile coordinates to pixel coordinates

        Returns a pair of floats
        """
        return (
                float(tile_coord[0] * self.cell_width),
                float(tile_coord[1] * self.cell_height),
            )

    def wall_at_pixel(self, pos):
        """True if there is a wall at the specified pixel coordinates
        """
        return self.wall_at_tile(self.pixel_to_tile(pos))

    def wall_at_tile(self, tile_coord):
        """True if there is a wall at the specified tile coordinates
        """
        try:
            tile_value = self.matrix[int(tile_coord[0]), int(tile_coord[1])]
        except IndexError:
            # Outside of the maze – a wall
            return True
        else:
            return tile_value in (-1, -2)

    def on_touch_down(self, touch):
        if not self.initialized:
            return
        for child in self.children[:]:
            # Pass the event to children
            if child.dispatch('on_touch_down', touch):
                return True
        if self.ball_source.collide_point(touch.x, touch.y):
            # Create a new ball
            ball = Ball(self, pos=self.tile_to_pixel((
                    (self.start_cranny[0] + 0.5),
                    (self.start_cranny[1] + 0.5),
                )))
            self.add_widget(ball)
            self.balls.append(ball)
            ball.on_touch_down(touch, force=True)
            self.parent.set_message(True,
                    "Move the ball to the opposite corner.")
        else:
            # Create a new builder
            tile_coord = self.pixel_to_tile(touch.pos)
            build_loop = BuildLoop(
                    spin=-1 if self.wall_at_tile(tile_coord) else 1,
                    pos=self.tile_to_pixel((
                        int(tile_coord[0]) + 0.5,
                        int(tile_coord[1]) + 0.5,
                    ))
                )
            self.add_widget(build_loop)
            build_loop.on_touch_down(touch, force=True)
            self.parent.set_message(False,
                    "Loop clockwise to build corridors, CCW to destroy them.")

    def win(self):
        """End the current round, destroy the widget.

        Called when a ball reaches the goal area.
        """
        for child in list(self.children):
            self.remove_widget(child)
        self.balls = []
        self.ball_source.size = 0, 0
        self.parent.win()

class MazeGame(Widget):
    """The top-level widget

    Keeps stats about the game, and the current MazeBoard
    """
    def __init__(self, **kwargs):
        numpy.random.seed(random.randint(0, 2 ** 30))
        super(MazeGame, self).__init__(**kwargs)
        # `times`: the elapsed times of the players, or “negative scores”
        self.times = [0, 0]
        # `current_solver`: index or the current solver (0 or 1)
        self.current_solver = 0
        # `round_number`: no. of urrent round; a round is 2 turns
        self.round_number = 0
        # `messages`: messages to the players
        self.messages = ['', '']

        # Start the game
        self.start()
        self.fully_drawn = False
        Clock.schedule_once(self.redraw)

    def set_message(self, for_solver, message):
        """Set the message to a player.

        If `for_solver` is True, sets the message to the solver, otherwise
        to their opponent.
        """
        self.messages[not for_solver ^ self.current_solver] = message
        self.redraw()

    def start(self):
        """Start a turn, by creating a MazeBoard
        """
        self.board = MazeBoard()
        self.add_widget(self.board)
        self.set_message(True, 'Wait...')
        self.set_message(False, 'Draw a loop to modify the maze.')

    def add_time(self, time):
        """Add some time th the current solver's clock
        """
        self.times[self.current_solver] += time
        self.redraw()

    def redraw(self, dt=None):
        """Redraw the whole screen
        """
        if not self.board.initialized:
            Clock.schedule_once(self.redraw)
            return
        if not self.fully_drawn:
            self.full_redraw()
        else:
            for edge in range(2):
                l = self.side_labels[edge]
                l.text = self.messages[edge]
                for player in range(2):
                    l = self.time_labels[edge][player]
                    l.text = time_format(self.times[player])

    tick = redraw

    def full_redraw(self):
        """Draw the game's chrome
        """
        self.reinit_before()
        self.time_labels = [[None, None], [None, None]]
        self.side_labels = [None, None]
        self.canvas.after.clear()
        with self.canvas.after:
            PopMatrix()
            PushMatrix()
            cell_size = self.board.cell_size
            for edge in range(2):
                # Messages at the non-player sides
                Color(0, 0, 0, 0.5)
                Rectangle(pos=(0, 0),
                        size=(self.board.window_width, cell_size))
                # Turn number
                turn_number = self.round_number * 2 + self.current_solver + 1
                num_turns = NUM_ROUNDS * 2
                if turn_number > num_turns:
                    text = ''
                else:
                    text = 'Turn {0}/{1}'.format(turn_number, num_turns)
                l = Label(text=text,
                        pos=(0, 0), size=(self.board.window_width, cell_size),
                        fontsize=cell_size)
                # Clocks
                for player in range(2):
                    Color(1, 1, 1, 1)
                    l = Label(text=time_format(self.times[player]),
                            fontsize=cell_size)
                    l.texture_update()
                    l.width = l.texture.width
                    if player ^ edge:
                        l.pos = cell_size, 0
                    else:
                        l.right = self.board.window_width - cell_size
                    l.height = self.board.cell_size
                    self.time_labels[edge][player] = l
                Translate(
                        self.board.window_width / 2,
                        self.board.window_height / 2,
                        0,
                    )
                Rotate(90, 0, 0, 1)
                # Messages to the players
                if edge ^ self.current_solver:
                    color = (0.7, 0.7, 0, 1)
                else:
                    color = (0, 0.5, 1, 1)
                l = Label(text=self.messages[edge],
                        pos=(
                                -self.board.window_height / 2,
                                -self.board.window_width / 2,
                            ),
                        size=(self.board.window_height, cell_size),
                        bold=True,
                        fontsize=cell_size,
                        color=color,
                    )
                self.side_labels[edge] = l
                if edge == 1:
                    break
                Rotate(90, 0, 0, 1)
                Translate(
                        -self.board.window_width / 2,
                        -self.board.window_height / 2,
                        0,
                    )
            PopMatrix()
        self.fully_drawn = True

    def reinit_before(self, base_angle=None):
        """Initialize the “before” canvas, i.e., whatever's below the MazeBoard
        """
        cell_size = self.board.cell_size
        if base_angle is None:
            base_angle = 180 if self.current_solver else 0
        self.canvas.before.clear()
        with self.canvas.before:
            for player, rotation in enumerate((90, 270)):
                # Win/lose messages at end of game
                if self.round_number < NUM_ROUNDS:
                    text = ''
                elif player ^ (self.times[0] < self.times[1]):
                    text = 'Winner!'
                else:
                    text = 'Second place'
                PushMatrix()
                Translate(
                        self.board.window_width / 2,
                        self.board.window_height / 2,
                        0,
                    )
                Rotate(rotation, 0, 0, 1)
                l = Label(text=text,
                        pos=(
                                -self.board.window_height / 2,
                                -self.board.window_width / 2 + cell_size,
                            ),
                        size=(self.board.window_height, cell_size * 4),
                        bold=True,
                        font_size=cell_size * 2,
                        color=(1, 1, 0),
                    )
                PopMatrix()
            PushMatrix()
            Translate(
                    self.board.window_width / 2,
                    self.board.window_height / 2,
                    0,
                )
            # Instructions for animating the board
            self.scale_instruction = Scale(1)
            self.rotate_instruction = Rotate(base_angle, 0, 0, 1)
            Translate(
                    -self.board.window_width / 2,
                    -self.board.window_height / 2,
                    0,
                )

    def win(self):
        """End one turn and start the next (if it wasn't the last turn
        """
        # Mostly animation code
        base_angle = 180 if self.current_solver else 0
        self.reinit_before(base_angle)
        animation = Animation(angle=base_angle + 90,
                t='out_cubic', duration=1)
        animation.start(self.rotate_instruction)
        animation = Animation(scale=0.01, t='out_cubic', duration=1)
        animation.start(self.scale_instruction)
        particle_shower(self.parent, type='win', pos=(
                self.board.window_width * self.current_solver,
                self.board.window_height * self.current_solver,
            ))
        old_board = self.board
        # Split the animation into a few functions: since generating the maze
        # takes some time, we do it when the MazeBoard is not visible, and
        # the animation to make it visible needs to start on the frame
        # after it's generated, otherwise the first part of the animation
        # would be effectively skipped.
        def next1(dt):
            if self.round_number < NUM_ROUNDS:
                self.start()
            self.remove_widget(old_board)
            Clock.schedule_once(next2)
        def next2(dt):
            Clock.schedule_once(next3)
        def next3(dt):
            if self.round_number < NUM_ROUNDS:
                self.full_redraw()
                self.reinit_before(base_angle + 90)
                animation = Animation(angle=base_angle + 180,
                        t='in_cubic', duration=1)
                animation.start(self.rotate_instruction)
                animation = Animation(scale=1, t='in_cubic', duration=1)
                self.scale_instruction.scale = 0.01
                animation.start(self.scale_instruction)
            else:
                self.set_message(True, '')
                self.set_message(False, '')
                self.full_redraw()
        if self.current_solver:
            self.round_number += 1
            self.current_solver = 0
        else:
            self.current_solver = 1
        Clock.schedule_once(next1, 1)

    def transform_touch(self, func, touch):
        """Transform a touch if the game board is rotated
        """
        touch.push()
        if self.current_solver and self.board and self.board.initialized:
            touch.x = self.board.window_width - touch.x
            touch.y = self.board.window_height - touch.y
        rv = func(touch)
        touch.pop()
        return rv

    def on_touch_down(self, touch):
        self.transform_touch(super(MazeGame, self).on_touch_down, touch)
        return True

    def on_touch_move(self, touch):
        self.transform_touch(super(MazeGame, self).on_touch_move, touch)
        return True

    def on_touch_up(self, touch):
        self.transform_touch(super(MazeGame, self).on_touch_up, touch)
        return True

def time_format(seconds):
    """Format a time in seconds for the clock display (mm:ss.ss)
    """
    return '{0:02.0f}:{1:05.2f}'.format(*divmod(seconds, 60))

def particle_shower(parent, type, pos):
    """Start a particle shower of the given type at the given position
    """
    rand = random.random
    if type == 'build':
        color_start = lambda i: (rand(), 1, rand(), 0)
        color_end = lambda i: (rand(), 1, rand(), 1)
        distance_start = lambda i: parent.cell_size * (rand() * 5 + 3)
        distance_end = lambda i: parent.cell_size / 2
        radius_start = lambda i: parent.cell_size
        radius_end = lambda i: 0
        duration = lambda i: rand() * 0.2 + 0.1
        source = lambda i: 'particle.png'
        number = 15
    elif type == 'destroy':
        color_start = lambda i: ((1, 1, 1, 1) if is_shard[i] else
                (1, rand(), rand(), 1))
        color_end = lambda i: ((1, 1, 1, 1) if is_shard[i] else
                (1, rand(), rand(), 0))
        distance_start = lambda i: parent.cell_size / 2
        distance_end = lambda i: parent.cell_size * (rand() * 5 + 5)
        radius_start = lambda i: parent.cell_size / 2 if is_shard[i] else 0
        radius_end = lambda i: 0 if is_shard[i] else parent.cell_size
        duration = lambda i: rand() * 0.4 + 0.1
        source = lambda i: (random.choice(['shard2.png', 'shard.png'])
                if is_shard[i] else 'particle.png')
        number = 15
        is_shard = [rand() < 0.2 for i in range(number)]
    elif type == 'win':
        window_size = sum(parent.get_parent_window().size) / 2
        color_start = lambda i: (rand(), rand(), 1, 1)
        color_end = lambda i: (rand(), rand(), 1, 0)
        distance_start = lambda i: 0
        distance_end = lambda i: rand() * window_size
        radius_start = lambda i: rand() * 2 * i
        radius_end = lambda i: rand() * i
        duration = lambda i: rand() * 0.5 + 0.5
        source = lambda i: 'particle2.png'
        number = 80
    else:
        return
    # If there is already a lot of particles on the parent, create less of them
    prev_number = getattr(parent, '_particles__num', 0) or 1
    number //= prev_number
    if number <= 0:
        number = 1
    parent._particles__num = prev_number + number
    # Create the particle widget
    w = TickingWidget()
    parent.add_widget(w)
    dur_max = 0
    with w.canvas:
        for i in range(number):
            dur = duration(i)
            dur_max = max(dur_max, dur)
            color = Color(*color_start(i))
            animation = Animation(rgba=color_end(i), duration=dur)
            animation.start(color)
            PushMatrix()
            dist = distance_start(i)
            angle = rand() * 2 * math.pi
            x = pos[0] + math.cos(angle) * dist
            y = pos[1] + math.sin(angle) * dist
            Translate(x, y, 0)
            translation = Translate()
            dist = distance_end(i)
            animation = Animation(xyz=(
                        pos[0] + math.cos(angle) * dist - x,
                        pos[1] + math.sin(angle) * dist - y,
                        0,
                    ),
                    duration=dur,
                )
            animation.start(translation)
            Rotate(rand() * 360, 0, 0, 1)
            particle = Point(pointsize=radius_start(i), source=source(i),
                points=[0, 0])
            animation = Animation(pointsize=radius_end(i), duration=dur)
            animation.start(particle)
            PopMatrix()
            def scope(color):
                def end_particle(dt):
                    color.a = 0
                    parent._particles__num -= 1
                Clock.schedule_once(end_particle, dur)
            scope(color)
    def die(dt):
        parent.remove_widget(w)
    Clock.schedule_once(die, dur_max)

def create_maze(width=81, height=51, complexity=0.975, density=0.975):
    """Create a random maze

    Adapted from: http://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * (shape[0] // 2 * shape[1] // 2))
    # Build actual maze
    Z = numpy.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make isles
    for i in range(density):
        x = random_integers(0, shape[1] // 2) * 2
        y = random_integers(0, shape[0] // 2) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[random_integers(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return Z

def calculate_points(x1, y1, x2, y2, spacing=5):
    """Calculate equally spaced points along a line

    Adapted from the Kivy "touchtracer" example
    """
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < spacing:
        return ()
    o = []
    m = dist / spacing
    for i in xrange(1, int(m)):
        mi = i / m
        lastx = x1 + dx * mi
        lasty = y1 + dy * mi
        o.append([lastx, lasty])
    return o

def schedule_tick(f):
    """Schedule a “tick” function to be called each frame
    """
    active = [True]
    def tick(dt):
        if not active:
            return
        f(dt)
        Clock.schedule_once(tick)
    def unschedule():
        del active[:]
        Clock.unschedule(tick)
    tick.unschedule = unschedule
    Clock.schedule_once(tick)
    return tick

class MazeApp(App):
    """The Kivy app for the maze game
    """
    def __init__(self, replay=None):
        super(MazeApp, self).__init__()

    def build(self):
        parent = Logger()
        parent.add_widget(MazeGame())

        return parent

if __name__ == '__main__':
    # re-importing this file so the classes are pickle-able
    # (otherwise they'd be pickled as  e.g. ‘__main__.MazeApp’ – useless)
    from touchgames.maze import MazeApp
    MazeApp().run()
