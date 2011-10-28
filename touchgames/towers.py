#! /usr/bin/env python
# Encoding: UTF-8

from __future__ import division

try:
    import numpy
    from numpy import pi, cos, sin, arctan2, sign
except ImportError:
    numpy = None
    from math import pi, cos, sin, arctan2
    from kivy.vector import Vector
    def sign(x):
        if x == 0:
            return 0
        return 1 if x > 0 else -1
import itertools
import random

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.graphics import (Color, Line, Rectangle, Triangle,
    Rotate, Translate, Scale, PushMatrix, PopMatrix)
from kivy.animation import Animation

from touchgames.mazesolver import solvemaze
from touchgames.util import FilledCircle, HollowCircle
from touchgames.replay import LoggedApp

# Tunable values:
CRITTER_SPEED = 4  # tiles per second
TOWER_PIXEL_SIZE = 40  # desired size of tower in pixels

# These are assumed by the graphics; don't change:
MAX_LEVEL = 6  # max. level of tower (see tower.draw() for markings)
TOWER_SIZE = 3  # tiles

def even(n):
    """Return an even number, either n, or n+1"""
    return n + n % 2

def clamp(value, minimum, maximum):
    """Clamp `value` to between `minimum` and `maximum`.

    Assumes minimum < maximum
    """
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
if numpy:
    ts = numpy.arange(num_circle_points + 1) * pi * 2 / num_circle_points
else:
    ts = [x * pi * 2 / num_circle_points for x in xrange(num_circle_points + 1)]
circle_points = list(roundrobin(
        ((cos(t) * 3, -sin(t) * 3) for t in ts),
        ((cos(t) * 2, -sin(t) * 2) for t in ts),
    ))

def RingSection(start, end, inner_radius, outer_radius):
    """Draw a section of a ring. Useful for circular “progressbars”.

    `start` and `end` are angles in radians
    `inner_radius` and `outer_radius` are in pixels
    The ring section is centered at (0, 0)
    """
    num = 64
    for t in range(int(start * num), int(end * num)):
        a = t * pi * 2 / num
        b = (t + 1) * pi * 2 / num
        Triangle(points=(
                cos(a) * inner_radius, sin(a) * inner_radius,
                cos(a) * outer_radius, sin(a) * outer_radius,
                cos(b) * inner_radius, sin(b) * inner_radius,
            ))
        Triangle(points=(
                cos(a) * outer_radius, sin(a) * outer_radius,
                cos(b) * inner_radius, sin(b) * inner_radius,
                cos(b) * outer_radius, sin(b) * outer_radius,
            ))

class Critter(Widget):
    """A Vicious Minion of Evil that is to be exterminated mercilessly
    """
    def __init__(self, direction, hp, **kwargs):
        """Initialize a Critter

        `direction`: direction in radians that the critter is facing initially
        `hp`: Health of the Critter
        """
        Widget.__init__(self, **kwargs)
        # Max health
        self.hp = hp
        # Direction we're facing currently
        self.direction = direction
        # Speed in tiles per second
        self.speed = CRITTER_SPEED
        # Damage done (accessed through .damage)
        self._damage = 0
        # Are we dead yet?
        self.dead = False
        # x-component of velocity
        self.xdir = int(sign(round(cos(direction))))
        # y-component of velocity
        self.ydir = int(sign(round(sin(direction))))
        # Initial velocity
        self.initial_dir = self.xdir, self.ydir

        self.draw()
        Clock.schedule_once(self.go)
        Clock.schedule_once(self.tick)

    @property
    def damage(self):
        return self._damage

    @damage.setter
    def damage(self, new_value):
        """Set damage, killing the critter if damaged enough
        """
        self._damage = new_value
        if new_value >= self.hp:
            self.die()
        self.set_color()

    def set_color(self):
        """Set color of the critter based on damage to it
        """
        red = self.damage / self.hp
        blue = 1 - self.damage / self.hp
        green = min(red, blue) / 2
        self.color_instruction.rgb = red, green, blue

    def draw(self):
        """Draw the critter as a circle with tiny eyes
        """
        self.canvas.clear()
        with self.canvas:
            PushMatrix()
            Translate(self.pos[0], self.pos[1], 0)
            # saved instructions are animated later:
            self.translate_instruction = Translate(0, 0, 0)
            Rotate(90 - self.direction * 180 / pi, 0, 0, 1)
            self.rotate_instruction = Rotate(0, 0, 0, 1)
            self.scale_instruction = Scale(1)
            self.color_instruction = Color(0, 0, 1)
            self.set_color()
            FilledCircle((0, 0), TOWER_PIXEL_SIZE / TOWER_SIZE * 4 / 5)
            Color(1, 1, 1)
            # Distance from center to an eye
            distance = TOWER_PIXEL_SIZE / TOWER_SIZE / 2
            # Draw eyes
            for angle in (-0.5, 0.5):
                Rectangle(pos=(
                            cos(angle) * distance - 1,
                            sin(angle) * distance - 1),
                        size=(2, 2))
            PopMatrix()

    def tick(self, dt):
        """Redraw the widget (called each frame since we're always moving)
        """
        if not self.parent:
            return

        self.canvas.ask_update()
        Clock.schedule_once(self.tick)

        # If the critter invaded a player's home area, eat some of the money
        if self.pos[0] <= TOWER_PIXEL_SIZE:
            self.eat_money(0, dt)
        elif self.pos[0] >= self.parent.window_width - TOWER_PIXEL_SIZE:
            self.eat_money(1, dt)

    def eat_money(self, side, dt):
        """Eat money from player at `side`, doing damage to self in the process
        """
        damage = min(self.hp - self.damage, self.hp * dt / 5)
        self.damage += damage
        self.parent.pay(side, damage * 10, True)

    def go(self, dt=0):
        """Figure out a new direction for this critter to go
        """
        if not self.parent:
            return

        # All calculations in pixels

        # List of candidate (x, y) offsets, equally spaced on a circle
        values = [(
                float(cos(t * pi / 20)) * self.parent.cell_width,
                float(sin(t * pi / 20)) * self.parent.cell_height)
            for t in range(0, 40)]
        # Change to (b, x, y), where b is the board matrix value at (x, y)
        values = [(self.parent.tile(self.pos[0] + x, self.pos[1] + y), x, y)
                for x, y in values]
        # Change to (c, x, y) where c is the “desirability” of movig to x, y
        # Also, remove points where (x, y) is in a tower/wall
        cs = self.parent.cell_size
        values = [(b +  # value of matrix
                # Randomize the direction somewhat:
                random.random() -
                # Prefer to keep moving in the same direction:
                (self.xdir * x + self.ydir * y) / 4 / (cs ** 2) -
                # Prefer to move in the initial direction (slightly)
                (self.initial_dir[0] * x + self.initial_dir[1] * y) / (cs ** 2),
                x, y,) for b, x, y in values if b >= -1]
        try:
            # Store the best offset to move to in (self.xdir, self.ydir)
            best, self.xdir, self.ydir = min(values)
        except ValueError:
            # No possible points – the critter is in a wall, such as at the
            # very start. Move in the original direction in this case
            self.xdir = self.initial_dir[0] * self.parent.cell_width
            self.ydir = self.initial_dir[1] * self.parent.cell_width

        # Time it'll take for this section
        time = 1 / self.speed

        # Update graphics & start animations
        x = self.xdir
        y = self.ydir
        self.draw()
        old_direcion = self.direction
        self.direction = arctan2(self.xdir, self.ydir)
        self.pos = self.pos[0] + x, self.pos[1] + y
        Animation(xy=(x, y), duration=time).start(self.translate_instruction)
        Animation(angle=(old_direcion - self.direction) * 180 / pi,
                duration=time / 2).start(self.rotate_instruction)

        # Prepare next time
        Clock.schedule_once(self.go, time)

    def die(self):
        """Die & remove this Critter
        """
        self.dead = True
        if not self.parent:
            return

        Clock.unschedule(self.go)
        # Death animation
        Animation(scale=10, duration=0.25).start(self.scale_instruction)
        Animation(a=0, duration=0.25).start(self.color_instruction)
        # Remove this widget after the animation
        def end(dt):
            if self.parent:
                self.parent.remove_widget(self)
        Clock.schedule_once(end, 0.25)

class Tower(Widget):
    """A Lofty Outpost of The Good Guys manned by valiant lasercannon engineers

    Each Tower has a level, and can be upgraded to the next level, or sold.
    Level 0 is a free “tower base” without a cannon, which must be upgraded
    immediately or it disappears when un-touched.
    Levels are indicated graphically by markings on the tower.
    """
    # Tower state summary:
    # touch_uid is None, upgrading_direction = 0: inactive
    # touch_uid is None, upgrading_direction = -1: un-upgrading
    # touch_uid is set, upgrading_direction = 0: touched, menu may be visible
    # touch_uid is set, upgrading_direction = 1: upgrading
    def __init__(self, side, **kwargs):
        super(Tower, self).__init__(**kwargs)
        # Index of the player this tower belongs to
        self.side = side
        # UID of the touch controlling the tower, if any
        self.touch_uid = None
        # State of an upgrading operation:
        #  -1 = cancelling an upgrade
        #   0 = no upgrade going on
        #   1 = upgrading
        # This is set to the desired value; i.e. if there are insufficient
        # while updgrading, upgrading_direction is 1 even though there's no
        # actual progress
        self.upgrading_direction = 0
        # Set if the upgrade/sell menu is visible
        self.menu_visible = False

        # Target this tower is currently firing at
        self.target = None

        Clock.schedule_once(self.shoot)

        # `direction`: Direction the cannon is facing, in radians
        if side:
            self.direction = pi
        else:
            self.direction = 0

        # Opacity of the laser shot (set to 1 with each salvo, then fades away)
        self.shot_opacity = 0

        # Tower stats (initially for level 0)
        self.level = 0
        self.power = 0  # damage to critter
        self.range = TOWER_SIZE  # in tiles
        self.interval = 1  # Hz
        self.upgrade_cost = 50  # €, to upgrade to next level
        self.build_speed = 100  # € per second

    def upgrade(self):
        """Upgrade the tower to the next level
        """
        self.level += 1
        self.power += 1
        self.range += TOWER_SIZE
        self.interval *= 0.7
        self.upgrade_cost *= 2

    @property
    def sell_cost(self):
        """Cost for selling this tower, in €
        """
        return self.sell_cost_for_level(self.level)

    @property
    def score_cost(self):
        cost = self.sell_cost
        if self.upgrading_direction:
            fract = self.upgrade_spent / self.upgrade_cost
            cost *= (1 - fract)
            cost += self.sell_cost_for_level(self.level + 1) * fract
        return cost

    def sell_cost_for_level(self, level):
        return {
                0: 0,
                1: 40, # €50 total cost (quite a good sell price)
                2: 100, # upgrades for €100, €150 total cost
                3: 200, # upgrades for €200, €350 total cost
                4: 400, # upgrades for €400, €750 total cost
                5: 800, # upgrades for €800,€1550 total cost
                6: 1600, # not obtainable
            }[level]

    @property
    def area(self):
        """Matrix slice index for the area this tower occupies
        """
        return (slice(self.coord[0], self.coord[0] + 3),
                slice(self.coord[1], self.coord[1] + 3))

    @property
    def zoc_area(self):
        """Matrix slice index for a slightly larger region than `area`

        Critters are deterred, but not prevented, from entering this region
        """
        return (slice(self.coord[0] - 1, self.coord[0] + 4),
                slice(self.coord[1] - 1, self.coord[1] + 4))

    def draw(self):
        """Draw the tower turret/markings/menu

        (The tower base is drawn in TowersGame as a background)
        """
        self.canvas.clear()
        if not self.parent:
            return

        with self.canvas:
            PushMatrix()
            # Center coordinate system at tower center, and rotate
            # so that 0 = away from player's home
            Translate(int(self.center[0]), int(self.center[1]), 0)
            if self.side:
                Rotate(180, 0, 0, 1)

            # Draw the tower markings
            if self.level > 1:
                Color(1, 1, 1, 0.9)
                w = self.parent.cell_width * 1.2
                h = self.parent.cell_height * 1.2
                if self.level == MAX_LEVEL:
                    # Level 6: big white rectangle
                    Rectangle(pos=(-w, -h), size=(2 * w, 2 * h))
                else:
                    # Levels 2 <= n <= 5: (n - 1) small triangles in corners
                    PushMatrix()
                    for i in range(self.level - 1):
                        Triangle(points=(w, h, w / 2, h, w, h / 2))
                        Rotate(90, 0, 0, 1)
                    PopMatrix()

            # If upgrading, draw a circular progress bar
            # showing upgrade_spent / upgrade_cost
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

                # If touched, visualize the tower's range
                if self.level:
                    center = self.center
                    Color(1, 1, 1, 0.1)
                    FilledCircle((0, 0), self.parent.cell_size * self.range)
                    Color(1, 1, 1, 0.2)
                    HollowCircle((0, 0), self.parent.cell_size * self.range)

                # The upgrade/sell menu
                if self.menu_visible:
                    show_sell = self.level and self.upgrade_touch_position < 1
                    show_upgrade = (self.level < MAX_LEVEL and
                            self.upgrade_touch_position > -1)

                    # Backgrounds
                    clamped_g = clamp(self.upgrade_touch_position, -1, 1)
                    if show_sell:
                        Color(0.75, 0.45, 0.35, 0.6 - clamped_g * 0.4)
                        RingSection(5 / 8, 7 / 8, self.parent.cell_size * 2.5,
                                self.parent.cell_size * 6.5)

                    if show_upgrade:
                        Color(0.45, 0.75, 0.35, 0.6 + clamped_g * 0.4)
                        RingSection(1 / 8, 3 / 8, self.parent.cell_size * 2.5,
                                self.parent.cell_size * 6.5)

                    # Labels
                    Color(0, 0, 0)
                    if show_upgrade:
                        # "Build" at level 0 since the tower's not really built
                        # yet (no money's been spent on it)
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
                                color=(0, 0, 0)
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
                                    color=(0, 0, 0)
                                )
            PopMatrix()
        self.draw_shot()

    def draw_shot(self, dt=0):
        """Draw the cannon and shot (in a layer above the level markings)
        """
        if not self.parent:
            return

        self.canvas.after.clear()
        with self.canvas.after:
            # Shot
            if self.target:
                # Fade out the shot dirung 1/4 the time beteen shots
                self.shot_opacity -= dt * self.interval * 4
                # Draw shot & schedule next draw
                if self.shot_opacity > 0:
                    Color(1, 0, 0, self.shot_opacity)
                    Line(points=itertools.chain(self.center, self.target.pos))
                    Clock.schedule_once(self.draw_shot)
                # Set cannon direction
                self.direction = arctan2(
                        self.target.pos[1] - self.center[1],
                        self.target.pos[0] - self.center[0],
                    )
            # Draw the cannon (if at least lv. 1)
            Color(0, 0, 0)
            if self.level:
                center = self.center
                direction = self.direction
                cell_size = self.parent.cell_size
                Line(points=itertools.chain(self.center, (
                        float(center[0] + cos(direction) * cell_size),
                        float(center[1] + sin(direction) * cell_size),
                    )))

    def upgrade_tick(self, dt):
        """Called each frame when upgrading (or cancelling upgrade)
        """
        if not self.parent:
            return

        # Upgrade “progress units” are €

        # Figure out amount of upgrade
        amount = self.upgrading_direction * dt * self.build_speed
        # Clamp to wha is needed for the tower
        amount = clamp(amount, -self.upgrade_spent,
                self.upgrade_cost - self.upgrade_spent)
        # Ask player to pay the corresponding amount (this might lower the
        # amount, or even make it negative!)
        amount = self.parent.pay(self.side, amount)
        self.upgrade_spent += amount
        if (self.upgrading_direction > 0 and
                self.upgrade_cost == self.upgrade_spent):
            self.upgrade()
            self.upgrading_direction = 0
            #self.touch_uid = None
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
            # Not upgrading yet: show the menu
            self.touch_uid = touch.uid
            self.menu_visible = True
            self.upgrading = False
            self.on_touch_move(touch)
        elif self.upgrading_direction < 0:
            # Cancelling upgrade: begin upgrading again
            self.touch_uid = touch.uid
            self.upgrading_direction = 1

    def on_touch_move(self, touch):
        # If this is the tower's controlling touch and the menu is shown, track
        # upgrade_touch_position
        if self.touch_uid == touch.uid:
            if self.menu_visible:
                self.upgrade_touch_position = touch.x - self.center[0]
                self.upgrade_touch_position /= self.parent.cell_width * 5
                if self.side:
                    # Menu is reversed for one player
                    self.upgrade_touch_position = -self.upgrade_touch_position
                if self.upgrade_touch_position > 1 and self.level < MAX_LEVEL:
                    # Far enough in the “Upgrade” section of menu:
                    # start upgrade!
                    self.upgrading_direction = 1
                    self.upgrade_spent = 0
                    self.menu_visible = False
                    Clock.unschedule(self.upgrade_tick)
                    Clock.schedule_once(self.upgrade_tick)
                # (selling requires touch_up)
                self.draw()

    def on_touch_up(self, touch):
        if self.touch_uid == touch.uid:
            if self.menu_visible and self.upgrade_touch_position < -1:
                # Far enough in the “Sell” section of menu: sell the tower
                self.sell()
            else:
                # Otherwise, remove menu & controlling touch,
                self.menu_visible = False
                self.touch_uid = None
                if self.upgrading_direction > 0:
                    # Cancel any upgrade
                    self.upgrading_direction = -1
                elif self.level == 0:
                    # Remove tower if it's the free lv.0 placeholder
                    self.parent.remove_tower(self)
                self.draw()

    def sell(self):
        """Sell the tower and remove it from play
        """
        parent = self.parent
        self.parent.remove_tower(self)
        parent.pay(self.side, -self.sell_cost, True)

    def shoot(self, dt=0):
        """Figure out what to shoot at and fire a laser ray at it
        """
        if not self.parent:
            return

        Clock.schedule_once(self.shoot, self.interval)

        if not self.level:
            return

        # Maximum squared distance to enemy
        dist_max = (self.range * self.parent.cell_size) ** 2
        if self.target:
            # I there's an existing target, keep shooting at it until it dies
            # or moves out of range
            if numpy:
                dist = sum((numpy.array(self.target.pos) - self.center) ** 2)
            else:
                dist = Vector(self.target.pos).distance2(self.center)
            if dist > dist_max or self.target.dead:
                self.target = None
        if not self.target:
            # If there's no target, pick one based on distance and its health
            # `possibilities`: (cost, target) tuples
            possibilities = []
            for target in self.parent.children:
                if isinstance(target, Critter) and not target.dead:
                    if numpy:
                        dist = sum((numpy.array(target.pos) - self.center) ** 2)
                    else:
                        dist = Vector(self.target.pos).distance2(self.center)
                    if dist < dist_max:
                        target_health = 1 - target.damage / target.hp
                        possibilities.append((dist * target_health, target))
            if not possibilities:
                # No possible target; bail
                self.target = None
                return
            else:
                # Select best target
                cost, self.target = min(possibilities)
        # Damage target
        self.target.damage += self.power
        self.shot_opacity = 1
        Clock.schedule_once(self.draw_shot)

        # If target died, collect the reward
        if self.target.dead:
            self.parent.pay(self.side, -self.target.hp * 2, True)

class TowersGame(Widget):
    """The main game widget
    """
    def __init__(self):
        self.initialized = False
        super(TowersGame, self).__init__()
        Clock.schedule_once(lambda dt: self.initialize())
        # Amount of money each player has, in €
        self.funds = [1000, 1000]
        # Towers currently in game
        self.towers = []
        # Color instructions for home bars
        self.home_bar_colors = {}

        # Critter release parameters
        self.critter_interval = 5
        self.critter_hp = 2

        # Side that won (0 or 1), if any
        self.winner = None

    def initialize(self):
        """Initialize the game
        """
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
        # - nonnegative numbers: free space; the value represents the shortest
        #   distance to the home area (honoring `costs`)
        # -1: a player's home area
        # -2: center wall
        # -3: tower
        self.matrix = numpy.zeros((
                    even(self.window_width // TOWER_PIXEL_SIZE) * TOWER_SIZE,
                    (self.window_height // TOWER_PIXEL_SIZE) * TOWER_SIZE,
                ), dtype=numpy.int8,
            )
        # `costs`: Movement cost for critters for each tile
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
        """Set a pair of Critters free
        """
        release_point = random.uniform(0, self.window_height)
        for direction in (0, pi):
            self.add_widget(Critter(
                    pos=(self.window_width // 2, release_point),
                    direction=direction,
                    hp=self.critter_hp,
                ))
        # Make the next generation of critters a bit tougher
        self.critter_hp += 1
        # After about 100 generations, HP starts increasing exponentially
        self.critter_hp = int(self.critter_hp * 1.01)
        # Also, lower the release interval
        self.critter_interval *= 0.98
        if self.critter_interval < 1:
            self.critter_interval = 1
        # Schedule the next wave
        Clock.schedule_once(self.release_critters, self.critter_interval)

    def setupMatrix(self):
        """Set up the wall and home areas in the matrix
        """
        self.matrix[:TOWER_SIZE, :] = -1
        self.matrix[-TOWER_SIZE:, :] = -1
        self.matrix[self.width // 2 - TOWER_SIZE + 1:
                self.width // 2 + TOWER_SIZE - 1, :] = -2

    def draw(self):
        """Draw the game board
        """
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
        """Draw labels (funds and win/lose message) the game board
        """
        self.canvas.after.clear()
        with self.canvas.after:
            PushMatrix()
            scores = self.funds[:]
            for tower in self.towers:
                scores[tower.side] += tower.score_cost
            total = sum(scores)
            if not total:
                # Avoid division by zero by cheating a bit
                total = 0.01
            '''
            # Score ball in the middle of screen
            if total > 0:
                position = scores[0] / total * self.window_height
                PushMatrix()
                Color(1, 1, 1, 0.5)
                Translate(self.window_width / 2, position, 0)
                RingSection(0, pi, 0, self.cell_width * .5)
                PopMatrix()
            '''

            Color(1, 1, 1, 0.5)
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
                        pos=(self.cell_size, 0),#(self.window_width - self.cell_size * 4) / 2),
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
                c_hp = self.critter_hp
                red = clamp(scores[side] / total * 2, 0, 1)
                if scores[side] < 0:
                    red = 0
                self.home_bar_colors[side].rgb = (1, clamp(red * 2, 0, 1), red)
            PopMatrix()

    def solvemaze(self):
        """“Solve” the matrix maze; return True if successful
        """
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
        """Pay `amount` € from player `side`'s funds. Amount may be negative

        `force`: if true, the whole `amount` is paid. If false, it is limited
        by the available funds.
        Returns amount that was actually paid.
        """
        if amount > self.funds[side] and not force:
            amount = self.funds[side]
        self.funds[side] -= amount
        self.draw_labels()

        # If the player is very short on money, confiscate some towers.
        # (very short means a debt equal to the HP of critters that are
        # currently being sent out — i.e. an increasing number)
        while self.funds[side] < -self.critter_hp:
            towers = [((abs(t.pos[0] - self.window_width / 2), t.level), t)
                    for t in self.towers if t.side == side and t.level]
            if towers:
                score, victim = min(towers)
                victim.sell()
            else:
                if self.winner is None:
                    self.winner = not side
                    self.draw()
                break

        return amount

    def on_touch_down(self, touch):
        """Either make a new tower, or display menu for an existing one
        """
        tower_coord = (touch.x // self.cell_width // TOWER_SIZE * TOWER_SIZE,
                touch.y // self.cell_height // TOWER_SIZE * TOWER_SIZE)
        matrix_coord = tower_coord[0] + 1, tower_coord[1] + 1
        if self.matrix[matrix_coord] < 0:
            # On tower?
            for tower in self.towers:
                # Yes, relay the touch to it
                if tower.coord == tower_coord:
                    tower.on_touch_down(touch)
                    break
            else:
                # Not on tower
                pass
        else:
            # On free space – make tower
            side = int(touch.x > self.window_width / 2)
            if self.funds[side] >= 0:
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
                else:
                    # Tower would block; display message to that effect
                    label = Label(
                            text='Blocking!',
                            pos=(-50, -50),
                            size=(100, 100),
                            font_size=self.cell_width * 2,
                            color=(1, 1, 1, 1),
                        )
                    self.add_widget(label)
                    with label.canvas.before:
                        PushMatrix()
                        Translate(touch.pos[0], touch.pos[1], 0)
                        Rotate(90 if side else 270, 0, 0, 1)
                    with label.canvas.after:
                        PopMatrix()
                    # Animate and fade out the message
                    anim = Animation(font_size=TOWER_PIXEL_SIZE * 2,
                            color=(1, 1, 1, 0), duration=1, t='in_quad')
                    anim.start(label)
                    def tick_blocking_anim(dt):
                        label.canvas.ask_update()
                        Clock.schedule_once(tick_blocking_anim)
                    tick_blocking_anim(0)
                    def end_blocking_anim(dt):
                        self.remove_widget(label)
                        Clock.unschedule(tick_blocking_anim)
                    Clock.schedule_once(end_blocking_anim, 1)

    def add_tower(self, tower):
        """Attempt to add the given tower. Returns True on success.
        """
        self.add_widget(tower)
        if self._set_towers(self.towers + [tower]):
            self.draw()
            return True
        else:
            self.remove_widget(tower)
            return False

    def remove_tower(self, tower):
        """Attempt to remove the given tower. Returns True on success.
        """
        self.costs[tower.zoc_area] = 1
        if self._set_towers([t for t in self.towers if t is not tower]):
            self.remove_widget(tower)
            self.draw()

    def _set_towers(self, new_towers):
        """Update `matrix` and `towers` to include the new towers.

        Returns True on success.
        """
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
        """Get the value of `matrix` at the given tile.

        Returns -3 (wall) for out-of-bounds tiles.
        """
        if x < 0 or y < 0:
            return -3
        try:
            return self.matrix[int(x) // self.cell_width,
                    int(y) // self.cell_height]
        except IndexError:
            return -3

class TowersApp(App):
    """The Kivy app for the towers game
    """
    def __init__(self, replay=None):
        super(TowersApp, self).__init__()

    def build(self):
        parent = Logger()
        parent.add_widget(TowersGame())

        return parent

if __name__ == '__main__':
    # re-importing this file so the classes are pickle-able
    # (otherwise they'd be pickled as  e.g. ‘__main__.TowersApp’ – useless)
    from touchgames.towers import TowersGame
    LoggedApp(TowersGame).run()
