# Encoding: UTF-8

from __future__ import division

import numpy
import itertools
import random

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.graphics import (Color, Ellipse, Line, Rectangle, Triangle, Point,
    Rotate, Translate, Scale, PushMatrix, PopMatrix, LineWidth)
from kivy.animation import Animation

from touchgames.mazesolver import solvemaze
from touchgames.util import FilledCircle, HollowCircle
from touchgames.replay import Logger

TOWER_PIXEL_SIZE = 30
TOWER_SIZE = 3  # tiles
MAX_LEVEL = 6

def even(n):
    """Return an even number, either n, or n+1"""
    return n + n % 2

def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))

num_circle_points = 100
ts = numpy.arange(num_circle_points + 1) * numpy.pi * 2 / num_circle_points
circle_points = list(roundrobin(
        ((numpy.cos(t) * 3, -numpy.sin(t) * 3) for t in ts),
        ((numpy.cos(t) * 2, -numpy.sin(t) * 2) for t in ts),
    ))

def RingSection(start, end, inner_radius, outer_radius):
    num = 64
    for t in range(int(start * num), int(end * num)):
        a = t * numpy.pi * 2 / num
        b = (t + 1) * numpy.pi * 2 / num
        Triangle(points=(
                numpy.cos(a) * inner_radius, numpy.sin(a) * inner_radius,
                numpy.cos(a) * outer_radius, numpy.sin(a) * outer_radius,
                numpy.cos(b) * inner_radius, numpy.sin(b) * inner_radius,
            ))
        Triangle(points=(
                numpy.cos(a) * outer_radius, numpy.sin(a) * outer_radius,
                numpy.cos(b) * inner_radius, numpy.sin(b) * inner_radius,
                numpy.cos(b) * outer_radius, numpy.sin(b) * outer_radius,
            ))

class Critter(Widget):
    def __init__(self, direction, hp, **kwargs):
        Widget.__init__(self, **kwargs)
        self.hp = hp
        self.direction = direction
        self.speed = 5
        self._damage = 0
        self.dead = False
        self.draw()
        Clock.schedule_once(self.go)
        Clock.schedule_once(self.tick)
        self.xdir = int(numpy.sign(round(numpy.cos(direction))))
        self.ydir = int(numpy.sign(round(numpy.sin(direction))))
        self.initial_dir = self.xdir, self.ydir

    @property
    def damage(self):
        return self._damage

    @damage.setter
    def damage(self, new_value):
        self._damage = new_value
        if new_value >= self.hp:
            self.die()
        self.set_color()

    def set_color(self):
        r = self.damage / self.hp
        b = 1 - self.damage / self.hp
        g = min(r, b) / 2
        self.color_instruction.rgb = r, g, b

    def draw(self):
        self.canvas.clear()
        with self.canvas:
            PushMatrix()
            Translate(self.pos[0], self.pos[1], 0)
            self.translate_instruction = Translate(0, 0, 0)
            Rotate(90 - self.direction * 180 / numpy.pi, 0, 0, 1)
            self.rotate_instruction = Rotate(0, 0, 0, 1)
            self.scale_instruction = Scale(1)
            self.color_instruction = Color(0, 0, 1)
            self.set_color()
            FilledCircle((0, 0), TOWER_PIXEL_SIZE / TOWER_SIZE * 4 / 5)
            Color(1, 1, 1)
            distance = TOWER_PIXEL_SIZE / TOWER_SIZE / 2
            for angle in (-0.5, 0.5):
                Rectangle(pos=(
                            numpy.cos(angle) * distance - 1,
                            numpy.sin(angle) * distance - 1),
                        size=(2, 2))
            PopMatrix()

    def tick(self, dt):
        if not self.parent:
            return

        self.canvas.ask_update()
        Clock.schedule_once(self.tick)

        if self.pos[0] <= TOWER_PIXEL_SIZE:
            side = 0
        elif self.pos[0] >= self.parent.window_width - TOWER_PIXEL_SIZE:
            side = 1
        else:
            return

        damage = min(self.hp - self.damage, self.hp * dt / 5)
        self.damage += damage
        self.parent.pay(side, damage * 10, True)

    def go(self, dt=0):
        if not self.parent:
            return

        values = [(
                float(numpy.cos(t * numpy.pi / 20)) * self.parent.cell_width,
                float(numpy.sin(t * numpy.pi / 20)) * self.parent.cell_height)
            for t in range(0, 40)]
        values = [(self.parent.tile(self.pos[0] + x, self.pos[1] + y), x, y)
                for x, y in values]
        cs = self.parent.cell_size
        values = [(b + random.random() -
                (self.xdir * x + self.ydir * y) / 4 / cs -
                (self.initial_dir[0] * x + self.initial_dir[1] * y) / 16 / cs,
                x, y,) for b, x, y in values if b >= -1]
        try:
            best, self.xdir, self.ydir = min(values)
        except ValueError:
            self.xdir = self.initial_dir[0] * self.parent.cell_width
            self.ydir = self.initial_dir[1] * self.parent.cell_width
        if self.xdir and self.ydir:
            time = 1.41421 / self.speed
        else:
            time = 1 / self.speed

        x = self.xdir
        y = self.ydir
        self.draw()
        old_direcion = self.direction
        self.direction = numpy.arctan2(self.xdir, self.ydir)
        self.pos = self.pos[0] + x, self.pos[1] + y
        Animation(xy=(x, y), duration=time).start(self.translate_instruction)
        Animation(angle=(old_direcion - self.direction) * 180 / numpy.pi,
                duration=time / 2).start(self.rotate_instruction)

        Clock.schedule_once(self.go, time)

    def die(self):
        self.dead = True
        if not self.parent:
            return

        Clock.unschedule(self.go)
        Animation(scale=10, duration=0.25).start(self.scale_instruction)
        Animation(a=0, duration=0.25).start(self.color_instruction)
        def end(dt):
            if self.parent:
                self.parent.remove_widget(self)
        Clock.schedule_once(end, 0.25)

class Tower(Widget):
    # States:
    # touch_uid is None, upgrading_direction = 0: inactive
    # touch_uid is None, upgrading_direction = -1: un-upgrading
    # touch_uid is set, upgrading_direction = 0: touched, menu may be visible
    # touch_uid is set, upgrading_direction = 1: upgrading
    def __init__(self, side, **kwargs):
        super(Tower, self).__init__(**kwargs)
        self.side = side
        self.touch_uid = None
        self.upgrading_direction = 0
        self.menu_visible = False

        self.target = None
        Clock.schedule_once(self.shoot)

        if side:
            self.direction = numpy.pi
        else:
            self.direction = 0

        self.level = 0
        self.power = 0
        self.range = 0.5 * TOWER_SIZE
        self.interval = 1
        self.upgrade_cost = 50
        self.build_speed = 100

    def upgrade(self):
        self.level += 1
        self.power += 1
        self.range += TOWER_SIZE
        self.interval *= 0.7
        self.upgrade_cost *= 2

    @property
    def sell_cost(self):
        if self.level == 1:
            return 40
        else:
            return self.upgrade_cost // 2

    @property
    def area(self):
        return (slice(self.coord[0], self.coord[0] + 3),
                slice(self.coord[1], self.coord[1] + 3))

    @property
    def zoc_area(self):
        return (slice(self.coord[0] - 1, self.coord[0] + 4),
                slice(self.coord[1] - 1, self.coord[1] + 4))

    def draw(self):
        self.canvas.clear()
        if not self.parent:
            return

        with self.canvas:
            PushMatrix()
            Translate(int(self.center[0]), int(self.center[1]), 0)

            if self.side:
                Rotate(180, 0, 0, 1)

            if self.level > 1:
                Color(1, 1, 1, 0.9)
                w = self.parent.cell_width * 1.2
                h = self.parent.cell_height * 1.2
                if self.level == MAX_LEVEL:
                    Rectangle(pos=(-w, -h), size=(2 * w, 2 * h))
                else:
                    PushMatrix()
                    for i in range(self.level - 1):
                        Triangle(points=(w, h, w / 2, h, w, h / 2))
                        Rotate(90, 0, 0, 1)
                    PopMatrix()

            if self.upgrading_direction:
                fract_done = self.upgrade_spent / self.upgrade_cost
                Color(0.35, 0.35, 0.35, 0.25)
                RingSection(fract_done, 1, self.parent.cell_size * 4,
                        self.parent.cell_size * 6)
                Color(0.35, 0.75, 0.35, 0.5)
                RingSection(0, fract_done,
                        self.parent.cell_size * 4,
                        self.parent.cell_size * 6)

            if self.touch_uid:
                Rotate(270, 0, 0, 1)
                if self.range:
                    center = self.center
                    Color(1, 1, 1, 0.1)
                    FilledCircle((0, 0), self.parent.cell_size * self.range)
                    Color(1, 1, 1, 0.2)
                    HollowCircle((0, 0), self.parent.cell_size * self.range)

                if self.menu_visible:
                    show_sell = self.level and self.upgrade_touch_position < 1
                    show_upgrade = (self.level < MAX_LEVEL and
                            self.upgrade_touch_position > -1)

                    clamped_g = clamp(self.upgrade_touch_position, -1, 1)
                    if show_sell:
                        Color(0.75, 0.45, 0.35, 0.6 - clamped_g * 0.4)
                        RingSection(5 / 8, 7 / 8, self.parent.cell_size * 2.5,
                                self.parent.cell_size * 6.5)

                    if show_upgrade:
                        Color(0.45, 0.75, 0.35, 0.6 + clamped_g * 0.4)
                        RingSection(1 / 8, 3 / 8, self.parent.cell_size * 2.5,
                                self.parent.cell_size * 6.5)

                    Color(0, 0, 0)
                    if show_upgrade:
                        if self.level:
                            text = 'Upgrade'
                        else:
                            text = 'Build'
                        Label(
                                text=u"%s\n€ %s" % (text, self.upgrade_cost),
                                pos=(-int(self.parent.cell_size * 2),
                                        int(self.parent.cell_size * 3)),
                                size=(int(self.parent.cell_size * 4),
                                        int(self.parent.cell_size * 3)),
                                halign='center',
                                font_size=10,
                            )
                    if show_sell:
                        if self.level:
                            Label(
                                    text=u"Sell\n€ %s" % self.sell_cost,
                                    pos=(-int(self.parent.cell_size * 2),
                                            -int(self.parent.cell_size * 6)),
                                    size=(int(self.parent.cell_size * 4),
                                            int(self.parent.cell_size * 3)),
                                    halign='center',
                                    font_size=10,
                                )
            PopMatrix()
        self.draw_shot()

    def draw_shot(self, dt=0):
        if not self.parent:
            return

        self.canvas.after.clear()
        with self.canvas.after:
            if self.target:
                self.shot_opacity -= dt * self.interval * 4
                if self.shot_opacity > 0:
                    Color(1, 0, 0, self.shot_opacity)
                    Line(points=itertools.chain(self.center, self.target.pos))
                    Clock.schedule_once(self.draw_shot)
                self.direction = numpy.arctan2(
                        self.target.pos[1] - self.center[1],
                        self.target.pos[0] - self.center[0],
                    )
            Color(0, 0, 0)
            if self.level:
                center = self.center
                direction = self.direction
                cell_size = self.parent.cell_size
                LineWidth(self.level)
                Line(points=itertools.chain(self.center, (
                        float(center[0] + numpy.cos(direction) * cell_size),
                        float(center[1] + numpy.sin(direction) * cell_size),
                    )))
                LineWidth(1)

    def upgrade_tick(self, dt):
        if not self.parent:
            return

        amount = self.upgrading_direction * dt * self.build_speed
        amount = clamp(amount, -self.upgrade_spent,
                self.upgrade_cost - self.upgrade_spent)
        amount = self.parent.pay(self.side, amount)
        self.upgrade_spent += amount
        if (self.upgrading_direction > 0 and
                self.upgrade_cost == self.upgrade_spent):
            self.upgrade()
            self.upgrading_direction = 0
            self.touch_uid = None
        elif self.upgrade_spent < 0 or (self.upgrade_spent == 0 and
                self.upgrading_direction < 0):
            self.parent.pay(self.side, -self.upgrade_spent)
            self.upgrading_direction = 0
            if self.level == 0:
                self.parent.remove_tower(self)
        else:
            Clock.schedule_once(self.upgrade_tick)
        self.draw()

    def on_touch_down(self, touch):
        if self.upgrading_direction == 0:
            self.touch_uid = touch.uid
            self.menu_visible = True
            self.upgrading = False
            self.on_touch_move(touch)
        elif self.upgrading_direction < 0:
            self.touch_uid = touch.uid
            self.upgrading_direction = 1

    def on_touch_move(self, touch):
        if self.touch_uid == touch.uid:
            if self.menu_visible:
                self.upgrade_touch_position = touch.x - self.center[0]
                self.upgrade_touch_position /= self.parent.cell_width * 5
                if self.side:
                    self.upgrade_touch_position = -self.upgrade_touch_position
                if self.upgrade_touch_position > 1 and self.level < MAX_LEVEL:
                    # Start upgrade!
                    self.upgrading_direction = 1
                    self.upgrade_spent = 0
                    self.menu_visible = False
                    Clock.unschedule(self.upgrade_tick)
                    Clock.schedule_once(self.upgrade_tick)
                self.draw()

    def on_touch_up(self, touch):
        if self.touch_uid == touch.uid:
            if self.menu_visible and self.upgrade_touch_position < -1:
                self.sell()
            else:
                self.menu_visible = False
                self.touch_uid = None
                if self.upgrading_direction > 0:
                    self.upgrading_direction = -1
                elif self.level == 0:
                    self.parent.remove_tower(self)
                self.draw()

    def sell(self):
        self.parent.pay(self.side, -self.sell_cost)
        self.parent.remove_tower(self)

    def shoot(self, dt=0):
        if not self.parent:
            return

        Clock.schedule_once(self.shoot, self.interval)

        dist_max = (self.range * self.parent.cell_size) ** 2
        if self.target:
            dist = sum((numpy.array(self.target.pos) - self.center) ** 2)
            if dist > dist_max or self.target.dead:
                self.target = None
        if not self.target:
            possibilities = []
            for target in self.parent.children:
                if isinstance(target, Critter) and not target.dead:
                    dist = sum((numpy.array(target.pos) - self.center) ** 2)
                    if dist < dist_max:
                        possibilities.append((dist, target))
            if not possibilities:
                self.target = None
                return
            else:
                dist, self.target = min(possibilities)
        self.target.damage += self.power
        self.shot_opacity = 1
        Clock.schedule_once(self.draw_shot)

        if self.target.dead:
            self.parent.pay(self.side, -self.target.hp * 2)

class TowersGame(Widget):
    def __init__(self):
        self.initialized = False
        super(TowersGame, self).__init__()
        Clock.schedule_once(lambda dt: self.initialize())
        self.must_redraw = False
        self.time = 0
        self.funds = [1000, 1000]
        self.towers = []
        self.home_bar_colors = {}

        self.critter_interval = 5
        self.critter_hp = 2

        self.winner = None

    def initialize(self):
        self.initialized = True

        win = self.get_parent_window()

        # `window_width` & `window_height`: dimensions of the window, in pixels
        self.window_width = win.width
        self.window_height = win.height

        self.matrix = numpy.zeros((
                    even(self.window_width // TOWER_PIXEL_SIZE) * TOWER_SIZE,
                    (self.window_height // TOWER_PIXEL_SIZE) * TOWER_SIZE,
                ), dtype=numpy.int8,
            )
        self.costs = self.matrix + 8
        # `width`/`height`: size of the maze, in tiles
        self.width, self.height = self.matrix.shape
        # `cell_width`/`cell_height`: size of a tile, in pixels
        self.cell_width = self.window_width / self.width
        self.cell_height = self.window_height / self.height
        # `cell_size`: average size of a tile, as a scalar
        self.cell_size = (self.cell_width + self.cell_height) / 2
        self.solvemaze()

        self.release_critters()

    def release_critters(self, dt=None):
        for direction in (0, numpy.pi):
            self.add_widget(Critter(
                    pos=(self.window_width // 2, self.window_height // 2),
                    direction=direction,
                    hp=self.critter_hp,
                ))
        self.critter_hp += 1
        self.critter_hp = int(self.critter_hp * 1.01)
        self.critter_interval *= 0.95
        if self.critter_interval < 1:
            self.critter_interval = 1
        Clock.schedule_once(self.release_critters, self.critter_interval)

    def setupMatrix(self):
        self.matrix[:TOWER_SIZE, :] = -1
        self.matrix[-TOWER_SIZE:, :] = -1
        self.matrix[self.width // 2 - TOWER_SIZE + 1:
                self.width // 2 + TOWER_SIZE - 1, :] = -2

    def draw(self):
        self.canvas.before.clear()
        with self.canvas.before:
            w = self.cell_width * TOWER_SIZE
            # Center bar
            Color(1, 1, 1, 0.5)
            Rectangle(pos=((self.window_width - w) / 2, 0),
                    size=(w, self.window_height))
            # Home bars
            for side, start_x in enumerate((0, self.window_width - w)):
                self.home_bar_colors[side] = Color(1, 1, 1)
                Rectangle(pos=(start_x, 0), size=(w, self.window_height))
            # Towers
            Color(0.5, 0.5, 0.5)
            for tower in self.towers:
                Rectangle(pos=tower.pos,
                    size=(w, self.cell_height * TOWER_SIZE))
        self.draw_labels()

    def draw_labels(self):
        self.canvas.after.clear()
        with self.canvas.after:
            PushMatrix()
            Translate(self.window_width, 0, 0)
            Rotate(90, 0, 0, 1)
            for side in reversed(range(2)):
                if not side:
                    Translate(self.window_height // 2,
                            self.window_width // 2, 0)
                    Rotate(180, 0, 0, 1)
                    Translate(-self.window_height // 2,
                            -self.window_width // 2, 0)
                Color(0, 0, 0)
                label = Label(text=u"€ %s" % int(round(self.funds[side])),
                        pos=(self.cell_size, 0),
                        size=(0, TOWER_PIXEL_SIZE),
                        font_size=self.cell_size,
                        color=(0, 0, 0),
                    )
                label.texture_update()
                label.width = label.texture.width
                if self.winner is not None:
                    if self.winner == side:
                        Color(1, 1, 0)
                        Label(text=u"Winner!",
                                pos=(0, self.cell_width * 5),
                                size=(self.window_height, self.cell_width * 5),
                                font_size=self.cell_size * 5,
                                color=(1, 1, 0),
                            )
                    else:
                        Color(1, 0, 0)
                        Label(text=u"Game over",
                                pos=(0, self.cell_width * 5),
                                size=(self.window_height, self.cell_width * 5),
                                font_size=self.cell_size * 5,
                                color=(1, 0, 0),
                            )
                red = clamp(-self.funds[side] / (self.critter_hp * 10), 0, 1)
                self.home_bar_colors[side].rgb = (1, 1 - red, 1 - red)
            PopMatrix()

    def solvemaze(self):
        self.setupMatrix()
        matrix = self.matrix
        mstart = numpy.zeros(matrix.shape + (1,), dtype=numpy.int32)
        mstart[matrix == -1] = 1
        corridors = matrix >= -1
        m = solvemaze(corridors, mstart, costs=self.costs)
        if m is None:
            return False
        m = m[:, :, 0]
        self.matrix = numpy.select([matrix < 0, True], [matrix, m])
        self.draw()
        return True

    def pay(self, side, amount, force=False):
        if amount > self.funds[side] and not force:
            amount = self.funds[side]
        self.funds[side] -= amount
        self.draw_labels()

        while self.funds[side] < -self.critter_hp * 10:
            towers = [((abs(t.pos[0] - self.window_width / 2), t.level), t)
                    for t in self.towers if t.side == side and t.level]
            if towers:
                score, victim = min(towers)
                victim.sell()
            else:
                Clock.unschedule(self.release_critters)
                if self.winner is None:
                    self.winner = not side
                    self.draw()
                break

        return amount

    def on_touch_down(self, touch):
        tower_coord = (touch.x // self.cell_width // TOWER_SIZE * TOWER_SIZE,
                touch.y // self.cell_height // TOWER_SIZE * TOWER_SIZE)
        matrix_coord = tower_coord[0] + 1, tower_coord[1] + 1
        if self.matrix[matrix_coord] < 0:
            # On tower?
            for tower in self.towers:
                if tower.coord == tower_coord:
                    tower.on_touch_down(touch)
                    break
            else:
                # Not on tower
                pass
        else:
            # On free space
            side = int(touch.x > self.window_width / 2)
            if self.funds[side] > 0:
                size = (self.cell_width * TOWER_SIZE,
                        self.cell_height * TOWER_SIZE)
                tower = Tower(
                        pos=(
                                touch.x // size[0] * size[0],
                                touch.y // size[1] * size[1]),
                        size=size,
                        side=side
                    )
                tower.coord = tower_coord
                if self.add_tower(tower):
                    tower.on_touch_down(touch)

    def add_tower(self, tower):
        self.add_widget(tower)
        if self.set_towers(self.towers + [tower]):
            self.draw()
            return True
        else:
            self.remove_widget(tower)
            return False

    def remove_tower(self, tower):
        self.costs[tower.zoc_area] = 1
        if self.set_towers([t for t in self.towers if t is not tower]):
            self.remove_widget(tower)
            self.draw()

    def set_towers(self, new_towers):
        oldm = self.matrix.copy()
        oldc = self.costs.copy()
        self.matrix = self.matrix * 0
        for t in new_towers:
            self.costs[t.zoc_area] = 100
            self.matrix[t.area] = -3
        self.setupMatrix()
        if self.solvemaze():
            self.towers = new_towers
            return True
        else:
            self.matrix = oldm
            self.costs = oldc
            return False

    def tile(self, x, y):
        if x < 0 or y < 0:
            return -3
        try:
            return self.matrix[int(x) // self.cell_width,
                    int(y) // self.cell_height]
        except IndexError:
            return -3

class TowersApp(App):
    def __init__(self, replay=None):
        super(TowersApp, self).__init__()

    def build(self):
        parent = Logger()
        parent.add_widget(TowersGame())

        return parent

if __name__ == '__main__':
    # re-importing this file so the classes are pickle-able
    # (otherwise they'd be pickled as  e.g. ‘__main__.TowersApp’ – useless)
    from touchgames.towers import TowersApp
    TowersApp().run()
