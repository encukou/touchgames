
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

from mazesolver import solvemaze

COUNTDOWN_START = 5
MAZE_CELL_SIZE = 30
BALL_SPEED = 10  # tiles per second
BALL_TOUCH_RADIUS = 1.5  # tiles
BALL_ZOC_RADIUS = 7  # tiles
MIN_LOOP_AREA = 3  # tiles squared
NUM_ROUNDS = 2

def yield_groups(source, n):
    return itertools.izip(*[iter(source)]*n)

def HollowCircle(pos, radius, segments=50):
    points = list(itertools.chain(*tuple((
                pos[0] + math.cos(t / segments * 2 * math.pi) * radius,
                pos[1] + math.sin(t / segments * 2 * math.pi) * radius,
            )
            for t in range(0, segments + 1)
        )))
    return Line(points=points)

def FilledCircle(pos=(0, 0), radius=1):
    return Ellipse(
            pos=(pos[0] - radius, pos[1] - radius),
            size=(radius * 2, radius * 2),
        )

class BallSource(Widget):
    def __init__(self, **kwargs):
        super(BallSource, self).__init__(**kwargs)

    def grow(self, size):
        animation = Animation(size=(size, size), t='in_cubic', duration=1)
        animation.start(self)
        tick = schedule_tick(self.tick)
        Clock.schedule_once(lambda dt: Clock.unschedule(tick), 2)

    def tick(self, dt):
        size = self.size[0]
        self.canvas.clear()
        with self.canvas:
            Color(0, 0, 1, 0.5)
            Ellipse(
                    pos=[p-s for p, s in zip(self.pos, self.size)],
                    size=[x * 2 - 1 for x in self.size],
                )
            Color(1, 1, 1, 1)
            HollowCircle(self.pos, self.size[0])

    def collide_point(self, x, y):
        px, py = self.pos
        sx, sy = self.size
        return (px - x) ** 2 + (py - y) ** 2 < sx ** 2

class TickingWidget(Widget):
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
        self.canvas.ask_update()

class Ball(TickingWidget):
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
        remaining_distance = dt * BALL_SPEED * self.parent.cell_size
        for i in range(50):
            if remaining_distance <= 0.01 or self.pos == self.target_pos:
                break
            distance_covered = self.move_step(remaining_distance)
            if distance_covered == 0:
                break
            remaining_distance -= distance_covered
        if self.translation_instruction.xy != self.pos:
            self.translation_instruction.xy = self.pos
            self.canvas.ask_update()
        if self.scale_instruction.scale != self.zoc_radius:
            self.scale_instruction.scale = self.zoc_radius
            self.canvas.ask_update()
        if not self.parent.ball_source.collide_point(*self.pos):
            self.parent.add_time(dt)
        if self.x < self.parent.cell_size:
            self.parent.win()

    def move_step(self, remaining_distance):
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
        # Now, we will deal with tile coordinates instead of pixels
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
            # elliptical in tile coordinates. (This isn't likely to matter 
            # anyway, though.)
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
    def __init__(self, build, **kwargs):
        super(Builder, self).__init__(**kwargs)
        self.points = [(self.pos[0], self.pos[1], True)]
        self.build = build
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
                    steps=3):
                ok = True
                for ball in self.parent.balls:
                    if ball.collide_zoc(x, y):
                        ok = False
                        break
                if not ok:
                    self.points.append((x, y, False))
                    continue
                else:
                    self.points.append((x, y, True))
                    tile_coord = parent.pixel_to_tile((x, y))
                    int_tile = int(tile_coord[0]), int(tile_coord[1])
                    if parent.set_wall(int_tile, self.build):
                        particle_shower(
                                parent=self.parent,
                                type='build' if self.build else 'destroy',
                                pos=self.parent.tile_to_pixel((
                                        int_tile[0] + 0.5,
                                        int_tile[1] + 0.5,
                                    )),
                            )
            max_points = self.max_points
            self.points = self.points[-max_points:]
            self.redraw()
            return True

    def on_touch_up(self, touch):
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

            animation = Animation(scale=self.parent.cell_size / 3 * 2,
                    t='out_cubic', duration=0.2)
            animation.start(self.scale_instruction)

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
            points = calculate_points(oldx, oldy, touch.x, touch.y, steps=3)
            if points:
                add_point = self.points.add_point
                for x, y in points:
                    add_point(x, y)

            signed_area = sum(
                    x1 * y2 - x2 * y1
                    for (x1, y1), (x2, y2)
                    in zip(
                            yield_groups(self.points.points, 2),
                            yield_groups(self.points.points[2:] +
                                self.points.points[:2], 2),
                        )
                ) / 2
            square_unit = self.parent.cell_width * self.parent.cell_height
            area_in_squares = signed_area / square_unit
            done = area_in_squares / MIN_LOOP_AREA
            if area_in_squares < 0:
                if done < -1:
                    done = -1
                color = 1 + done, 1, 0
            else:
                if done > 1:
                    done = 1
                color = 1, 1 - done, 0
            self.color_instruction.rgb = color
            if abs(done) == 1:
                x, y = self.parent.pixel_to_tile(self.pos)
                end_x, end_y = self.parent.pixel_to_tile(touch.pos)
                if int(x) == int(end_x) and int(y) == int(end_y):
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
        animation = Animation(scale=self.parent.cell_size * 15,
                t='in_cubic', duration=0.15)
        animation.start(self.scale_instruction)

        animation = Animation(a=0, duration=0.15)
        animation.start(self.color_instruction)

        def _die(dt):
            if self.parent:
                self.parent.remove_widget(self)
        Clock.schedule_once(_die, 0.15)

    def tick(self, dt):
        self.canvas.ask_update()

class MazeBoard(TickingWidget):
    darkWalls = True

    def __init__(self):
        self.initialized = False
        super(MazeBoard, self).__init__()
        Clock.schedule_once(lambda dt: self.initialize())
        self.must_redraw = False
        self.time = 0

    def initialize(self):
        self.initialized = True

        win = self.get_parent_window()
        self.window_width = win.width
        self.window_height = win.height

        self.matrix = -numpy.transpose(
                numpy.array(create_maze(
                        self.window_width // MAZE_CELL_SIZE,
                        self.window_height // MAZE_CELL_SIZE,
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

        self.balls = []

        self.start_point = self.width - 2, self.height - 2
        self.start_cranny = self.width - 1, self.height - 2

        self.bdist = self.dist = self.matrix
        self.set_walls(self.matrix)

        self.background_canvas = Canvas()
        self.canvas.add(self.background_canvas)
        self.draw()

        self.ball_source = BallSource(size=(0, 0),
                pos=(self.window_width, self.window_height))
        self.add_widget(self.ball_source)

        Clock.schedule_once(lambda dt: self.countdown(COUNTDOWN_START), 1)
        Clock.schedule_interval(self.redraw, 0.1)

    def countdown(self, num):
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
        self.time += time
        if self.time > 15:
            self.parent.set_message(True,
                    u"Take another ball if you're really stuck.")
        self.parent.add_time(time)

    def draw(self):
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
        self.must_redraw = True

    def tick(self, dt=None):
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
        if self.matrix[x, y] == -4:
            return self.tile_color(x - 1, y)
        elif self.matrix[x, y] == -3:
            return self.tile_color(x + 1, y)
        elif self.matrix[x, y] < 0:
            return 0, 0, 0, 0
        else:
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
                return numpy.array((r, g, b, 1)) ** 8

    def set_walls(self, matrix):
        """Set walls and solve the maze. No-op if maze is unsolvable
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

        self.build_tries = defaultdict(set)
        return True

    def set_wall(self, coord, create):
        if coord in self.build_tries[create]:
            return False
        self.build_tries[create].add(coord)
        if coord == self.start_cranny or coord == (2, 1):
            return False
        m = self.matrix.copy()
        if not create and self.matrix[coord] >= 0:
            m[coord] = -1
            rv = self.set_walls(m)
        elif create and self.matrix[coord] == -1:
            m[coord] = 0
            rv = self.set_walls(m)
        else:
            return False
        if rv:
            self.redraw()
        return rv

    def pixel_to_tile(self, pos):
        return (
                pos[0] / self.cell_width,
                pos[1] / self.cell_height,
            )

    def tile_to_pixel(self, tile_coord):
        return (
                float(tile_coord[0] * self.cell_width),
                float(tile_coord[1] * self.cell_height),
            )

    def wall_at_pixel(self, pos):
        return self.wall_at_tile(self.pixel_to_tile(pos))

    def wall_at_tile(self, tile_coord):
        try:
            tile_value = self.matrix[int(tile_coord[0]), int(tile_coord[1])]
        except IndexError:
            return True
        else:
            return tile_value in (-1, -2)

    def on_touch_down(self, touch):
        for child in self.children[:]:
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
        for child in list(self.children):
            self.remove_widget(child)
        self.balls = []
        self.ball_source.size = 0, 0
        self.parent.win()

class MazeGame(Widget):
    def __init__(self, **kwargs):
        super(MazeGame, self).__init__(**kwargs)
        self.times = [0, 0]
        self.current_solver = 0
        self.round_number = 0
        self.messages = ['', '']
        self.start()
        self.fully_drawn = False
        Clock.schedule_once(self.redraw, 1)

    def set_message(self, for_solver, message):
        self.messages[not for_solver ^ self.current_solver] = message
        self.redraw()

    def start(self):
        self.board = MazeBoard()
        self.add_widget(self.board)
        self.set_message(True, 'Wait...')
        self.set_message(False, 'Draw a loop to modify the maze.')

    def add_time(self, time):
        self.times[self.current_solver] += time
        self.redraw()

    def redraw(self, dt=None):
        if not self.board.initialized:
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
        self.reinit_before()
        self.time_labels = [[None, None], [None, None]]
        self.side_labels = [None, None]
        self.canvas.after.clear()
        with self.canvas.after:
            PopMatrix()
            PushMatrix()
            cell_size = self.board.cell_size
            for edge in range(2):
                Color(0, 0, 0, 0.5)
                Rectangle(pos=(0, 0),
                        size=(self.board.window_width, cell_size))
                turn_number = self.round_number * 2 + self.current_solver + 1
                num_turns = NUM_ROUNDS * 2
                if turn_number > num_turns:
                    text = ''
                else:
                    text = 'Turn {0}/{1}'.format(turn_number, num_turns)
                l = Label(text=text,
                        pos=(0, 0), size=(self.board.window_width, cell_size),
                        fontsize=cell_size)
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
        cell_size = self.board.cell_size
        if base_angle is None:
            base_angle = 180 if self.current_solver else 0
        self.canvas.before.clear()
        with self.canvas.before:
            for player, rotation in enumerate((90, 270)):
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
            self.scale_instruction = Scale(1)
            self.rotate_instruction = Rotate(base_angle, 0, 0, 1)
            Translate(
                    -self.board.window_width / 2,
                    -self.board.window_height / 2,
                    0,
                )

    def win(self):
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

    def super_touch(self, func, touch):
        touch.push()
        if self.current_solver and self.board and self.board.initialized:
            touch.x = self.board.window_width - touch.x
            touch.y = self.board.window_height - touch.y
        rv = func(touch)
        touch.pop()
        return rv

    def on_touch_down(self, touch):
        self.super_touch(super(MazeGame, self).on_touch_down, touch)
        return True

    def on_touch_move(self, touch):
        self.super_touch(super(MazeGame, self).on_touch_move, touch)
        return True

    def on_touch_up(self, touch):
        self.super_touch(super(MazeGame, self).on_touch_up, touch)
        return True

def time_format(seconds):
    return '{0:02.0f}:{1:05.2f}'.format(*divmod(seconds, 60))

def particle_shower(parent, type, pos):
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
    shape = ((height//2)*2+1, (width//2)*2+1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity*(5*(shape[0]+shape[1])))
    density    = int(density*(shape[0]//2*shape[1]//2))
    # Build actual maze
    Z = numpy.zeros(shape, dtype=bool)
    # Fill borders
    Z[0,:] = Z[-1,:] = 1
    Z[:,0] = Z[:,-1] = 1
    # Make isles
    for i in range(density):
        x = random_integers(0,shape[1]//2)*2
        y = random_integers(0,shape[0]//2)*2
        Z[y,x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append( (y,x-2) )
            if x < shape[1]-2:
                neighbours.append( (y,x+2) )
            if y > 1:
                neighbours.append( (y-2,x) )
            if y < shape[0]-2:
                neighbours.append( (y+2,x) )
            if len(neighbours):
                y_,x_ = neighbours[random_integers(0,len(neighbours)-1)]
                if Z[y_,x_] == 0:
                    Z[y_,x_] = 1
                    Z[y_+(y-y_)//2, x_+(x-x_)//2] = 1
                    x, y = x_, y_
    return Z

def calculate_points(x1, y1, x2, y2, steps=5):
    # Adapted from the Kivy "touchtracer" example
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < steps:
        return ()
    o = []
    m = dist / steps
    for i in xrange(1, int(m)):
        mi = i / m
        lastx = x1 + dx * mi
        lasty = y1 + dy * mi
        o.append([lastx, lasty])
    return o

def schedule_tick(f):
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
    def build(self):
        parent = Widget()
        parent.add_widget(MazeGame())

        return parent


if __name__ == '__main__':
    MazeApp().run()
