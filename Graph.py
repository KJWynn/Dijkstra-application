from MinHeap import MinHeap
class Graph:
    """

    """
    def __init__(self, vertices, start, end, last_vertex):
        self.vertices = vertices
        self.last_vertex = last_vertex
        self.num_vertices = len(self.vertices)
        self.heap = MinHeap()
        self.start = start
        self.end = end
        self.cost = float("inf")


        # initialize heap
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            if i == self.start:
                vertex.distance = 0
            self.heap.add(vertex)  # O(log L)

    def __str__(self):
        return_string = ""
        for vertex in self.vertices:
            return_string += str(vertex) + "\n"
        return return_string


class Vertex:
    """

    """
    def __init__(self, item):
        self.discovered = False
        self.visited = False
        self.distance = float("inf")
        self.item = item
        self.index = None
        self.edges = []
        self.previous = None
        self.has_passenger = False
        self.has_passenger_along = False


    def __str__(self):
        if self.has_passenger:
            _string = "Special "
        else:
            _string = ""
        return_str = _string + str(self.item) + " with distance " + str(self.distance) + " and position in heap: " + str(self.index)
        return return_str

    def get_edges(self):
        return_str = "Vertex " + str(self.item) + " has the following edges:"
        for edge in self.edges:
            return_str += str(edge) + " "
        return return_str


class Edge:
    """

    """
    def __init__(self, v: Vertex, w1: int, w2: int):
        self.v = v
        self.actual_weight = None
        self.w1 = w1
        self.w2 = w2
        self.difference = w1-w2

    def __str__(self):
        return_string = ""
        return_string += f"({self.v.item}, {self.w1}, {self.w2})"
        return return_string