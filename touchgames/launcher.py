#! /usr/bin/env python


import sys

def main(gamename, *args):
    module = __import__('touchgames.games.%s' % gamename, fromlist=['main'])
    print module
    module.main(*args)

def script_entry():
    try:
        gamename = [a for a in sys.argv if a[0] != '-'][1]
    except IndexError:
        print "Specify a game to play."
        exit(1)
    args = sys.argv[2:]
    main(gamename, *args)

if __name__ == '__main__':
    script_entry()