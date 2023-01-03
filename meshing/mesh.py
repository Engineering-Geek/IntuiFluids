import numpy as np
from typing import Any, Dict, Callable, Tuple, Union, List
from tqdm.auto import tqdm
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiPoint, Point
from scipy.spatial import Delaunay
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt


class Coordinate:
    def __init__(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z
        self.point = Point(x, y, z)
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
    
    @property
    def z(self):
        return self._z
    
    @property
    def coordinates(self):
        return (self.x, self.y, self.z)
    
    def __call__(self):
        return self.coordinates


class Node:
    def __init__(self, coordinate: Coordinate, id: int, phi_values: Dict[str, float], node_type: str):
        self._coordinate = coordinate
        self.phi_values = phi_values
        self.id = id
        self.neighbors = []
        self.node_type = node_type
    
    @property
    def x(self):
        return self.coordinate.x

    @property
    def y(self):
        return self.coordinate.y
    
    @property
    def z(self):
        return self.coordinate.z
    
    @property
    def coordinates(self):
        return self._coordinate.coordinates
    
    def distance(self, x: float, y: float, z: float) -> float:
        return np.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2)
    
    def __str__(self):
        return f"Node {self.id}: ({self.x}, {self.y}, {self.z})"
    
    def __repr__(self):
        return f"Node {self.id}: ({self.x}, {self.y}, {self.z})"
    
    def append_neighbor(self, neighbor) -> bool:
        if neighbor is not None and neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            return True
        return False
    
    def remove_neighbor(self, neighbor) -> bool:
        if neighbor in self.neighbors:
            self.neighbors.remove(neighbor)
            return True
        return False
    

class BoundarySurface:
    """
    A boundary surface is a collection of nodes that are on the boundary of the problem. If the problem is 2d, then the boundary surface is a line. If the problem is 3d, then the boundary surface is a plane.
    """
    def __init__(self, coordinates: List[Node], boundary_type: str, phi_values: Dict[str, float], verbose: bool = False):

        self.coordinates = coordinates
        self.boundary_type = boundary_type
        self.phi_values = phi_values
        self.verbose = verbose
        self.nodes = self._generate_nodes()
    
    def _generate_nodes(self) -> List[Node]:
        """
        This function generates the nodes on the boundary surface. It is called by the constructor.
        
        This works by first finding the unique x, y, and z coordinates of the nodes on the boundary surface. 
        Then, it creates a tensor of nodes with the same dimensions as the unique x, y, and z coordinates. 
        The tensor is then filled with nodes in the respective places. 
        
        The nodes are then assigned neighbors by looping through the tensor and checking if the nodes in the tensor are neighbors.
        
        Total runtime is O(n^3) where n is the number of nodes on the boundary surface.
        """
        unique_x = np.unique([node.x for node in self.coordinates])
        unique_y = np.unique([node.y for node in self.coordinates])
        unique_z = np.unique([node.z for node in self.coordinates])
        # order the coordinates
        unique_x.sort()
        unique_y.sort()
        unique_z.sort()
        
        node_tensor = np.empty((len(unique_x), len(unique_y), len(unique_z)), dtype=Node)
        node_id = 0
        
        x_iter = tqdm(unique_x, desc="Generating nodes", disable=(not self.verbose))
        for i, x in enumerate(unique_x):
            for j, y in enumerate(unique_y):
                for k, z in enumerate(unique_z):
                    node_tensor[i, j, k] = Node(Coordinate(x, y, z), node_id, self.phi_values, self.boundary_type)
                    node_id += 1
        
        # assign neighbors
        for i, x in enumerate(unique_x):
            for j, y in enumerate(unique_y):
                for k, z in enumerate(unique_z):
                    node = node_tensor[i, j, k]
                    if i > 0:
                        node.append_neighbor(node_tensor[i-1, j, k])
                    if i < len(unique_x) - 1:
                        node.append_neighbor(node_tensor[i+1, j, k])
                    if j > 0:
                        node.append_neighbor(node_tensor[i, j-1, k])
                    if j < len(unique_y) - 1:
                        node.append_neighbor(node_tensor[i, j+1, k])
                    if k > 0:
                        node.append_neighbor(node_tensor[i, j, k-1])
                    if k < len(unique_z) - 1:
                        node.append_neighbor(node_tensor[i, j, k+1])
        
        nodes = [node for node in node_tensor.flatten() if node is not None]
        return nodes
    
    def __str__(self):
        return f"BoundarySurface: {self.boundary_type}"
    


class Mesh:
    """
    # Mesh
    The mesh is a collection of nodes that are connected to each other (like a data graph).
    
    This is an abstract class that is inherited by the 2d and 3d mesh classes to solve various problems.
    """
    def __init__(self, boundaries: Dict[BoundarySurface], verbose: bool = False, initial_phi_values: Dict[str, float] = None):        
        self.verbose = verbose
        self._nodes: List[Node] = self._generate_nodes(boundaries, verbose, initial_phi_values)
            
    def _generate_nodes(self, boundaries, initial_phi_values) -> List[Node]:
        """
        This function generates the nodes in the mesh. It is called by the constructor.
        
        This works by first finding the unique x, y, and z coordinates of the nodes in the mesh. 
        Then, it creates a tensor of nodes with the same dimensions as the unique x, y, and z coordinates. 
        The tensor is then filled with nodes in the respective places. 
        
        The nodes are then assigned neighbors by looping through the tensor and checking if the nodes in the tensor are neighbors.
        
        Total runtime is O(n^3) where n is the number of nodes in the mesh.
        """
        unique_x = np.unique([node.x for boundary in boundaries.values() for node in boundary.nodes])
        unique_y = np.unique([node.y for boundary in boundaries.values() for node in boundary.nodes])
        unique_z = np.unique([node.z for boundary in boundaries.values() for node in boundary.nodes])
        # order the coordinates
        unique_x.sort()
        unique_y.sort()
        unique_z.sort()
        
        node_tensor = np.empty((len(unique_x), len(unique_y), len(unique_z)), dtype=Node)
        node_id = 0
        
        # insert boundary nodes into the node tensor
        boundaries_iter = tqdm(boundaries.values(), desc="Inserting boundary nodes", disable=(not self.verbose))
        for boundary in boundaries_iter:
            for node in boundary.nodes:
                i = np.where(unique_x == node.x)[0][0]
                j = np.where(unique_y == node.y)[0][0]
                k = np.where(unique_z == node.z)[0][0]
                node_tensor[i, j, k] = node
                node_tensor[i, j, k].node_id = node_id
                node_id += 1
        
        # Make a 3d point cloud of the nodes using shapely
        point_cloud = MultiPoint([node.point for node in node_tensor.flatten() if node is not None])
        triangulation = Delaunay(point_cloud)
        
        # fill in the rest of the nodes as interior nodes
        x_iter = tqdm(unique_x, desc="Generating interior nodes", disable=(not self.verbose))
        for i, x in enumerate(x_iter):
            for j, y in enumerate(unique_y):
                for k, z in enumerate(unique_z):
                    if node_tensor[i, j, k] is None and triangulation.find_simplex((x, y, z)) >= 0:
                        node_tensor[i, j, k] = Node(Coordinate(x, y, z), node_id, initial_phi_values, "Interior")
                        node_id += 1
        
        # assign neighbors
        x_iter = tqdm(unique_x, desc="Assigning neighbors", disable=(not self.verbose))
        for i, x in enumerate(x_iter):
            for j, y in enumerate(unique_y):
                for k, z in enumerate(unique_z):
                    node = node_tensor[i, j, k]
                    if node is not None:
                        if i > 0:
                            node.append_neighbor(node_tensor[i-1, j, k])
                        if i < len(unique_x) - 1:
                            node.append_neighbor(node_tensor[i+1, j, k])
                        if j > 0:
                            node.append_neighbor(node_tensor[i, j-1, k])
                        if j < len(unique_y) - 1:
                            node.append_neighbor(node_tensor[i, j+1, k])
                        if k > 0:
                            node.append_neighbor(node_tensor[i, j, k-1])
                        if k < len(unique_z) - 1:
                            node.append_neighbor(node_tensor[i, j, k+1])
        
        # Turn the node tensor into a list of nodes and remove the None values
        nodes = [node for node in node_tensor.flatten() if node is not None]
        
        # sort the nodes by node_id
        nodes.sort(key=lambda node: node.node_id)
        return nodes
    
    @abstractmethod
    def solve(self, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = False) -> np.ndarray:
        pass
    
    @abstractmethod
    def generate_matrices(self) -> Tuple[np.ndarray[(Any, Any)], np.ndarray]:
        pass
    
    def visualize(self, phi: str, save_path: str = None):
        # if the mesh is 2d, then plot the phi values on a contour plot
        if all([node.z == 0 for node in self._nodes]):
            x = np.unique([node.x for node in self._nodes])
            y = np.unique([node.y for node in self._nodes])
            z = np.zeros((len(x), len(y)))
            for node in self._nodes:
                i = np.where(x == node.x)[0][0]
                j = np.where(y == node.y)[0][0]
                z[i, j] = node.phi[phi]
            plt.contourf(x, y, z)
            plt.colorbar()
            plt.title(f"{phi} values")
            if save_path is not None:
                plt.savefig(save_path)
            plt.show()
        else:
            # if the mesh is 3d, then plot the phi values on a 3d scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = [node.x for node in self._nodes]
            y = [node.y for node in self._nodes]
            z = [node.z for node in self._nodes]
            c = [node.phi[phi] for node in self._nodes]
            ax.scatter(x, y, z, c=c)
            ax.set_title(f"{phi} values")
            if save_path is not None:
                plt.savefig(save_path)
            plt.show()
    
    def __str__(self):
        return f"Mesh: {len(self._nodes)} nodes"
    
    def __len__(self):
        return len(self._nodes)
    
    def __getitem__(self, index):
        return self._nodes[index]
    
    @property
    def nodes(self) -> List[Node]:
        return self._nodes
        
    @property
    def phi_names(self) -> List[str]:
        return [phi for phi in self._nodes[0].phi_values.keys()]

class FiniteVolumeMesh(Mesh):
    def __init__(self, boundaries: Dict[str, Boundary], initial_phi_values: Dict[str, float], verbose: bool = False):
        super().__init__(boundaries, initial_phi_values, verbose)
        self._A, self._b = self._generate_matrices()
    
    def _generate_matrices(self) -> Tuple[np.ndarray[(Any, Any)], np.ndarray]:
        # generate the A matrix
        A = np.zeros((len(self._nodes), len(self._nodes)))
        for i, node in enumerate(self._nodes):
            if node.node_type == "Interior":
                A[i, i] = -sum([node.get_face_area(neighbor) for neighbor in node.neighbors])
                for neighbor in node.neighbors:
                    A[i, neighbor.node_id] = node.get_face_area(neighbor)
        
        # generate the b matrix
        b = np.zeros((len(self._nodes), 1))
        for i, node in enumerate(self._nodes):
            b[i] = sum([node.get_face_area(neighbor) * neighbor.phi_values["phi"] for neighbor in node.neighbors])
        
        return A, b
    
    def solve(self, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = False) -> np.ndarray:
        # solve the system of linear equations
        phi_values = np.linalg.solve(self._A, self._b)
        
        # update the phi values of the nodes
        for i, node in enumerate(self._nodes):
            node.phi_values = phi_values[i]
        
        return phi_values


