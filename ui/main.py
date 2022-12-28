from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.gridlayout import GridLayout
from kivymd.uix.boxlayout import BoxLayout
from kivymd.uix.navigationdrawer import MDNavigationDrawer, MDNavigationDrawerMenu
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.stacklayout import MDStackLayout


class HomeStackLayout(MDStackLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding = 10
        self.spacing = 10
        self.add_widget(MDRectangleFlatButton(text="Simulation 1"))
        self.add_widget(MDRectangleFlatButton(text="Simulation 2"))
        self.add_widget(MDRectangleFlatButton(text="Simulation 3"))


class SimulationScreen(Screen):
    pass


class HomeScreen(Screen):
    pass


class SimulationApp(MDApp):
    def build(self):
        self.load_kv("Simulation.kv")


if __name__=="__main__":
    SimulationApp().run()


