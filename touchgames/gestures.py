
import math
import numpy

def normalize_angle(angle):
    while angle > math.pi:
        angle -= math.pi * 2
    while angle <= -math.pi:
        angle += math.pi * 2
    return angle

def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def pinch_helper(a, b, finger_size):
    if a['uid'] == b['uid']:
        return 0, None, None
    a_start = numpy.array(a['history'][:2][-1])
    a_end = numpy.array(a['history'][-2:][0])
    b_start = numpy.array(b['history'][:2][-1])
    b_end = numpy.array(b['history'][-2:][0])

    # Opposing lines?
    if abs(normalize_angle(a['direction'] - b['direction'])) > math.pi * 3 / 4:
        # Pinch gesture
        dir2 = math.atan2(a_end[1] - b_end[1], a_end[0] - b_end[0])
        if abs(normalize_angle(a['direction'] - dir2)) > math.pi * 3 / 4:
            distance_between_ends = distance(a_end, b_end)
            if distance_between_ends < 4 * finger_size:
                midpoint = (a_end + b_end) / 2
                return -distance_between_ends, 'pinch', tuple(midpoint)
        # Unpinch gesture
        dir2 = math.atan2(a_start[1] - b_start[1], a_start[0] - b_start[0])
        if (abs(normalize_angle(math.pi + a['direction'] - dir2)) >
                math.pi * 3 / 4):
            distance_between_starts = distance(a_start, b_start)
            if distance_between_starts < 4 * finger_size:
                midpoint = (a_start + b_start) / 2
                return -distance_between_starts, 'unpinch', tuple(midpoint)
    return 0, None, None

class GestureHelper(object):
    def __init__(self, finger_size):
        self.current_touches = dict()
        self.finger_size = finger_size

    def touch_down(self, touch):
        d = self.current_touches[touch.uid] = dict(
                history=[],
                mean=touch.pos,
                bounding_box=touch.pos + touch.pos,
                direction=(0, 0),
                done=False,
                uid=touch.uid,
            )
        d['concurrent_touches'] = dict(self.current_touches)
        for other_touch in self.current_touches.values():
            if other_touch['uid'] != touch.uid:
                other_touch['concurrent_touches'][touch.uid] = d
        return self.touch_move(touch)

    def touch_up(self, touch):
        if touch.uid not in self.current_touches:
            return
        self.current_touches[touch.uid]['done'] = True
        rv = self.touch_move(touch)
        del self.current_touches[touch.uid]
        return rv

    def touch_move(self, touch):
        if touch.uid not in self.current_touches:
            return
        d = self.current_touches[touch.uid]
        history = d['history']
        history.append(touch.pos)
        start = history[0]
        end = history[-1]
        d['mean'] = tuple(
                d['mean'][i] + (touch.pos[i] - d['mean'][i]) / len(history)
                for i in range(2))
        d['bounding_box'] = (
                min(d['bounding_box'][0], touch.x),
                min(d['bounding_box'][1], touch.y),
                max(d['bounding_box'][2], touch.x),
                max(d['bounding_box'][3], touch.y),
            )
        d['direction'] = math.atan2(history[-1][1] - history[0][1],
            history[-1][0] - history[0][0])
        #d['medoid'] = min(history, key=lambda (x, y):
        #        abs(x - d['mean'][0]) + abs(y - d['mean'][1]))

        path_size = (d['bounding_box'][2] - d['bounding_box'][0] +
                d['bounding_box'][3] - d['bounding_box'][1])
        if path_size < self.finger_size:
            d['type'] = 'point'
        else:
            d['type'] = 'line'
        (pinch_score, pinch_type, midpoint), other = min(
                (pinch_helper(d, c, self.finger_size), c)
                for c in d['concurrent_touches'].values())
        if pinch_type:
            d['pinch_type'] = pinch_type
            d['pinch_buddy'] = other
            d['pinch_target'] = midpoint
        else:
            d['pinch_type'] = None
        touch.ud['gesture_data'] = d

        return d
