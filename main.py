from kivy.app import App

from touchgames.maze import MazeGame

class MazeApp(App):
    def build(self):
        return MazeGame()

MazeApp().run()
