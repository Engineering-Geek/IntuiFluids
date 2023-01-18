from kivymd.app import MDApp
from kivymd.uix.floatlayout import FloatLayout
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.graphics import Canvas, Color, Line
import numpy as np


class SimulationBoxLayout(MDBoxLayout):
    def draw(self, x_values: np.array, boundary_layer_thickness: np.array, thermal_boundary_layer_thickness: np.array):
        # scale the x_values to the width of the box
        # scale the boundary_layer_thickness or thermal_boundary_layer_thickness to the height of the box, whichever is bigger
        
        box_x_values = x_values * self.width / max(x_values)
        # max_thickness = max(max(boundary_layer_thickness), max(thermal_boundary_layer_thickness))
        max_thickness = 10.0
        box_boundary_layer_thickness = boundary_layer_thickness * self.height / max_thickness
        box_thermal_boundary_layer_thickness = thermal_boundary_layer_thickness * self.height / max_thickness

        thermal_layer_points = []
        boundary_layer_points = []
        for x, boundary_layer, thermal_layer in zip(box_x_values, box_boundary_layer_thickness, box_thermal_boundary_layer_thickness):
            thermal_layer_points.append(x)
            thermal_layer_points.append(thermal_layer)
            boundary_layer_points.append(x)
            boundary_layer_points.append(boundary_layer)

        # draw the boundary layer thickness in red and the thermal boundary layer thickness in blue
        self.canvas.clear()
        with self.canvas:
            Color(0, 0, 1)
            Line(points=boundary_layer_points)
            Color(1, 0, 0)
            Line(points=thermal_layer_points)


class PrandtleSimulation(FloatLayout):
    nu = 1.0
    alpha = 1.0
    length = 0.25
    height = 0.075
    u = 1.0
    pr = 1.0
    def update_pr(self, pr):
        self.pr = pr
        self.run_simulation()
    
    def update_nu(self, nu):
        self.nu = nu
        self.run_simulation()
    
    def update_u(self, u):
        self.u = u
        self.run_simulation()

    def _boundary_layer_thickness(self, x, u):
        return 5 * np.sqrt(self.nu * x / u)
    
    def _thermal_boundary_layer_thickness(self, x, u, pr):
        if pr == 0:
            return np.array([0.0 for _ in x])
        return (pr ** (-1/3.0)) * self._boundary_layer_thickness(x, u)
    
    def run_simulation(self):
        pr = self.pr
        nu = self.nu
        u = self.u
        simulation_box = self.ids['simulation_box']

        x_values = np.linspace(0, self.length, 101)
        boundary_layer_thickness = self._boundary_layer_thickness(x_values, u)
        thermal_boundary_layer_thickness = self._thermal_boundary_layer_thickness(x_values, u, pr)

        simulation_box.draw(x_values, boundary_layer_thickness, thermal_boundary_layer_thickness)

        self.ids['pr'].text = f'Pr: {self.pr:.2f}'



class PrandtleApp(MDApp):
    def build(self):
        return PrandtleSimulation()


if __name__ == '__main__':
    PrandtleApp().run()
