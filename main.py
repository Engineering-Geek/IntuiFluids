# from __future__ import annotations

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.card import MDCard
from kivy.properties import StringProperty


class SimulationCard(MDCard):
    text = StringProperty()
    image_path = StringProperty()


class MainWindow(MDScreen):
    pass


class SecondWindow(MDScreen):
    pass


class ThirdWindow(MDScreen):
    pass


class WindowManager(MDScreenManager):
    pass


class MainApp(MDApp):
    def build(self):
        self.theme_cls.material_style = "M3"
        self.theme_cls.primary_palette = "BlueGray"
        self.theme_cls.primary_hue = "500"
        return WindowManager()
    
    def on_start(self):
        self.root.current = "main"


if __name__ == "__main__":
    MainApp().run()