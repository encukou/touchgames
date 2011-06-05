# Encoding: UTF-8

import cPickle as pickle
import sys
from datetime import datetime
import gzip
import random
from time import time

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
import kivy.clock
from kivy.input.motionevent import MotionEvent

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
        fileobj = open(log_filename, 'wb')
        self.stream = gzip.GzipFile(fileobj=fileobj)
        self.log('random', random.getstate())

    def tick(self, dt):
        """The base class implementation redraws the widget's canvas.
        """
        self.log('dt', dt)
        Clock.schedule_once(self.tick)

    def add_widget(self, child):
        super(Logger, self).add_widget(child)
        self.log('add_widget', type(child))

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

class ReplayMotionEvent(MotionEvent):
    """A dummy MotionEvent, since kivy won't let us use MotionEvent directly
    """
    pass

class Replay(App):
    """A widget that replays a game from a log file captured by Logger

    Replay monkeypatches kivy.clock.time to fire events read from the log
    file.

    This is more of a dirty hack than other parts of the code.
    """
    def __init__(self, log_filename):
        super(Replay, self).__init__()
        kivy.clock.time = self.time
        self.stream = gzip.GzipFile(fileobj=open(log_filename, 'rb'))
        self.start_time = time()
        self.stream_time = 0
        self.next_id = 0

    @property
    def time_elapsed(self):
        return time() - self.start_time

    def build(self):
        self.top_widget = Widget()
        return self.top_widget

    def time(self):
        """Like time.time(), but handles events from the log file
        """
        while self.stream and self.time_elapsed > self.stream_time:
            rec = pickle.load(self.stream)
            getattr(self, 'handle_' + rec[0])(*rec[1:])
        return time()

    # Handlers for events from the log file:

    def handle_random(self, state):
        """Set the random state"""
        random.setstate(state)

    def handle_add_widget(self, cls):
        """Add a child widget"""
        self.top_widget.add_widget(cls())

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
        self.top_widget.dispatch(event, touch)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: %s <logfile>" % sys.argv[0]
    else:
        Replay(sys.argv[1]).run()

