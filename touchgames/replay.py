#! /usr/bin/env python
# Encoding: UTF-8

from __future__ import division

import cPickle as pickle
import sys
from datetime import datetime
import gzip
import random
from time import time

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.graphics import Rectangle, Color, Scale, PushMatrix, PopMatrix
import kivy.clock
from kivy.input.motionevent import MotionEvent

class LoggedApp(App):
    def __init__(self, widget_class):
        super(LoggedApp, self).__init__()
        self.widget_class = widget_class

    def build(self):
        self.logger = parent = Logger()
        parent.add_widget(self.widget_class())

        return parent

    def run(self):
        super(LoggedApp, self).run()
        self.logger.close()

class Logger(Widget):
    """A widget that logs time, touches and other info into a file

    Things logged are:
    - classes of child widgets
    - random state
    - touch down/move/up
    - lock ticks and intervals

    In short, everything that needs to be logged to replay a game is logged.

    There are lots of limitations (e.g. any arguments to child widgets or
    classes of touches are not logged), but for these games it works.
    """
    def __init__(self, **kwargs):
        super(Logger, self).__init__(**kwargs)
        Clock.schedule_once(self.tick)

        log_filename = 'log-%s.log' % datetime.now().isoformat()
        self._fileobj = fileobj = open(log_filename, 'wb')
        self.stream = gzip.GzipFile(fileobj=fileobj)
        self.log('random', random.getstate())
        self.have_size = False

    def tick(self, dt):
        """The base class implementation redraws the widget's canvas.
        """
        if not self.have_size:
            win = self.get_parent_window()
            self.log('window_size', win.width, win.height)
            self.have_size = True
        self.log('dt', dt)
        Clock.schedule_once(self.tick)

    def add_widget(self, child):
        super(Logger, self).add_widget(child)
        self.log('add_widget', type(child))
        self.log('random', random.getstate())
        self.have_size = False

    def on_touch_down(self, touch):
        self.log_touch('down', touch)
        super(Logger, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        self.log_touch('move', touch)
        super(Logger, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        self.log_touch('up', touch)
        super(Logger, self).on_touch_up(touch)

    def log_touch(self, action, touch):
        touch_attrs = {}
        for attr in set(touch.__attrs__) | set(['uid']):
            touch_attrs[attr] = getattr(touch, attr)
        self.log(action, touch_attrs)

    def log(self, *args):
        pickle.dump(args, self.stream, protocol=2)
        self.stream.flush()

    def close(self):
        self.stream.close()
        self._fileobj.close()

class ReplayMotionEvent(MotionEvent):
    """A dummy MotionEvent, since kivy won't let us use MotionEvent directly
    """
    pass

class Replay(App):
    """A widget that replays a game from a log file captured by Logger

    Replay monkeypatches kivy.clock.time to fire events read from the log
    file.

    Touching the replay will change the replay speed based on the y (up/down)
    -coordinate of the touch, from 50% to 200%. (NB. every clock tick is still
    processed, so speedups might not actually be that big).

    This is more of a dirty hack than other parts of the code.
    """
    def __init__(self, log_filename):
        super(Replay, self).__init__()
        self.log_filename = log_filename
        kivy.clock.time = self.time
        self.stream = gzip.GzipFile(fileobj=open(self.log_filename, 'rb'))
        self.first_tick = time()
        self.last_tick = time()
        self.time_elapsed = 0
        self.stream_time = 0
        self.next_id = 0
        self.clock_speed = 1
        self.running = True

    def build(self):
        self.top_widget = Widget()
        self.content_widget = Widget()
        self.top_widget.add_widget(self.content_widget)
        self.top_widget.add_widget(SpeedAdjuster(self))
        return self.top_widget

    def time(self):
        """Like time.time(), but handles events from the log file
        """
        current = time()
        self.time_elapsed += (current - self.last_tick) * self.clock_speed
        self.last_tick = current
        if not self.running:
            return self.last_tick
        while self.stream and self.time_elapsed > self.stream_time:
            try:
                rec = pickle.load(self.stream)
            except EOFError:
                self.handle__end()
                self.running = False
                break
            else:
                getattr(self, 'handle_' + rec[0])(*rec[1:])
                if rec[0] == 'dt':
                    # Ensure every tick gets handled
                    break
        return self.first_tick + self.stream_time

    def handle__end(self):
        parent = self.top_widget.get_parent_window()
        with self.top_widget.canvas:
            Color(0, 0, 0.2, 0.5)
            Rectangle(size=(parent.width, 40),
                    pos=(0, parent.height / 2 - 20))
        self.top_widget.add_widget(Label(text='Replay finished',
                width=self.top_widget.get_parent_window().width,
                pos=(0, parent.height / 2 - 5), height=10, color=(1, 1, 1, 1)))

    # Handlers for events from the log file:

    def handle_random(self, state):
        """Set the random state"""
        random.setstate(state)

    def handle_add_widget(self, cls):
        """Add a child widget"""
        self.content_widget.add_widget(cls())

    def handle_dt(self, dt):
        """Tick the clock"""
        self.stream_time += dt

    def handle_down(self, attrs):
        """Generate a touch-down event"""
        self.handle__touch_event(attrs, 'on_touch_down')

    def handle_move(self, attrs):
        """Generate a touch-move event"""
        self.handle__touch_event(attrs, 'on_touch_move')

    def handle_up(self, attrs):
        """Generate a touch-up event"""
        self.handle__touch_event(attrs, 'on_touch_up')

    def handle__touch_event(self, attrs, event):
        """Generate any touch-down event"""
        touch = ReplayMotionEvent(None, attrs['id'], attrs)
        for name, value in attrs.items():
            setattr(touch, name, value)
        self.content_widget.dispatch(event, touch)

    def handle_window_size(self, width, height):
        print width, height
        children_to_do = set([self.content_widget])
        children_done = set()
        while children_to_do:
            child = children_to_do.pop()
            child.window_width = width
            child.window_height = height
            children_done.add(child)
            children_to_do.update(child.children)
            children_to_do.difference_update(children_done)
        self.content_widget.canvas.before.clear()
        with self.content_widget.canvas.before:
            PushMatrix()
            win = self.content_widget.get_parent_window()
            window_width = win.width
            window_height = win.height
            the_min = min(window_width / width, window_height / height)
            Scale(the_min)
        self.content_widget.canvas.after.clear()
        with self.content_widget.canvas.after:
            PopMatrix()

class SpeedAdjuster(Widget):
    """Widget that adjusts the clock_speed of a Replay,
    based on its last touch's y-coordinate
    """
    def __init__(self, replay):
        self.replay = replay
        self.active_touch_uid = None
        super(SpeedAdjuster, self).__init__()

    def on_touch_down(self, touch):
        try:
            self.remove_widget(self.display)
        except AttributeError:
            pass
        self.active_touch_uid = touch.uid
        return self.on_touch_move(touch)

    def on_touch_move(self, touch):
        if self.active_touch_uid == touch.uid:
            self.replay.clock_speed = factor = (touch.sy * 1.91 + 0.1)
            self.canvas.clear()
            with self.canvas:
                Color(0.2, 0, 0, 0.5)
                Rectangle(size=(self.get_parent_window().width, 20),
                        pos=(0, touch.y - 10))
            self.display = Label(text='Replay speed: {0:03}%'.format(
                    int(factor * 100)), width=self.get_parent_window().width,
                    pos=(0, touch.y - 5), height=10, color=(1, 1, 1, 1))
            self.add_widget(self.display)
        return True

    def on_touch_up(self, touch):
        if self.active_touch_uid == touch.uid:
            self.canvas.clear()
            self.remove_widget(self.display)
        return True

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: %s <logfile>" % sys.argv[0]
    else:
        Replay(sys.argv[1]).run()
