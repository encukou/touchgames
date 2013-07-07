from kivy.app import App

from touchgames.maze import MazeGame

class MazeApp(App):
    def build(self):
        return MazeGame()

   def on_pause(self):
      # No important data to save
      return True

   def on_resume(self):
      pass

MazeApp().run()
