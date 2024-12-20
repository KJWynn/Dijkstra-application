from __future__ import annotations

class MinHeap:
    """
    Modified from provided FIT1008 implementation by Brendon Taylor and Jackson Goerner

    Min Heap implementation using a Python in-built list.
    Parent node's index is (child node's index - 1) // 2.
    Left child's index = 2 * node_idx + 1, right child index = 2 * node_idx + 2
    Cost of comparison between integers is assumed O(1) and not shown in function complexity

    Attributes:
        self.array: The array to store items using a Python list

    """

    def __init__(self, size):
        """
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.array = [None for i in range(size)]

    def print_heap(self):
        print("\n")
        for vertex in self.array:
            print(vertex)

    def rise(self, added_item_idx):
        """
        Rise the added item to its correct position

        :Input: 
            added_item_idx: The item's index in the heap array that might need to rise
        :Post condition: Heap ordering is observed
        :Time complexity: O(log N), where N is the number of nodes in the heap
        :Aux space complexity: O(1)
        """

        added_item = self.array[added_item_idx]
        while added_item_idx > 0 and self.array[added_item_idx].distance <self.array[(added_item_idx - 1) // 2].distance:
            # swap added element with its parent if added element is smaller than its parent node
            self.array[added_item_idx], self.array[(added_item_idx - 1) // 2] = self.array[(added_item_idx - 1) // 2], \
                                                                                self.array[added_item_idx]
            # also update index(location in heap array)
            self.array[added_item_idx].index, self.array[(added_item_idx - 1) // 2].index = self.array[(added_item_idx - 1) // 2].index, self.array[
                                                                                                added_item_idx].index
            # update swapped element's index
            added_item_idx = (added_item_idx - 1) // 2

        self.array[added_item_idx] = added_item

    def sink(self, outofplace_item_idx):
        """
        Sink the out-of-place item to its correct position. Also updates index of item(position in heap)

        :Input:
            outofplace_item_idx: Index of item that was swapped to the root during the add operation
        :Post condition: Heap ordering is observed
        :Time complexity: O(log N), where N is the number of nodes in the heap
        :Aux space complexity: O(1)
        """
        # save the item
        outofplace_item = self.array[outofplace_item_idx]
        saved_smallestchild_index = int()

        # get the final position for the outofplace item
        while 2 * outofplace_item_idx + 2 <= len(self.array):  # while there is child nodes to check
            smallest_idx = self.smallest_child(outofplace_item_idx)
            if self.array[smallest_idx].distance >= outofplace_item.distance:
                break

            saved_oop_index = outofplace_item_idx # 0 4
            saved_smallestchild_index = smallest_idx # 1

            self.array[outofplace_item_idx] = self.array[smallest_idx]  # rise the smallest child
            
            # update index of the child that rises
            self.array[outofplace_item_idx].index = saved_oop_index

            outofplace_item_idx = smallest_idx

            # place outofplace item into hole
        self.array[outofplace_item_idx] = outofplace_item
        # update index of outofplace item
        self.array[outofplace_item_idx].index = saved_smallestchild_index

    def smallest_child(self, item_idx):
        """
        Returns the index of item's child that has the smaller value.

        :Input:
            item_idx: Index of item whose smallest child we want to find
        :Precondition: 1 <= item_idx <= len(self.array) // 2 - 1
        :Return: Index of smallest child
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        left_idx = 2 * item_idx + 1
        right_idx = 2 * item_idx + 2
        if left_idx == len(self.array) - 1 or \
                self.array[left_idx].distance < self.array[right_idx].distance:
            return left_idx
        else:
            return right_idx

    def add(self, item):
        """
        Add an item into the heap, then rises the item to its correct postion
        
        :Input:
            item: Item to be added to heap
        :Post condition: Heap ordering is observed
        :Time complexity: O(log N), where N is the number of nodes in the heap

        """
        self.array[item.index] = item # O(1)
        self.rise(item.index) # O(log N)




    def getMin(self):
        """
        Replaces minimum item(root) with last item, then sinks last item into its correct position, returns minimum item.
        
        :Return: The minimum item in the heap(root)
        :Time complexity: O(log N), where N is the number of nodes in the heap
        :Aux space complexity: O(1)
        """
        if len(self.array) == 0:
            return None
        elif len(self.array) == 1:
            return self.array.pop()
        else:
            # save minimum item
            min_item = self.array[0]

            # put last item into minimum item's place
            last_item = self.array.pop()
            last_item.index = 0
            self.array[0] = last_item

            # last item may be out of place, so sink it to correct position, O(log N)
            self.sink(0)

            return min_item


class Graph:
    """
    Graph data structure. Stores a list of vertices, the number of vertices, V(last vertex+1). The vertices are from 0..num_vertices-1
    """
    def __init__(self, vertices):
        """
        Initialize graph with vertices.

        :Input:
            vertices: List of vertices
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        
        """
        self.vertices = vertices
        self.num_vertices = len(self.vertices)
        self.cost = float("inf")

    def __str__(self):
        return_string = ""
        for vertex in self.vertices:
            return_string += str(vertex) + "\n"
        return return_string


class Vertex:
    """
    Vertex class representing a vertex. Each vertex has a list of outgoing edges, represented by a list of Edge objects 
    The item attribute is the shorthand way of representing the Vertex

    """
    def __init__(self, item):
        """
        Initialize the vertex. Set distance to infinity
        
        :Input:
            item: Integer representing the vertex. 
        :Time complexity: O(1)
        :Space complexity: O(1)
        
        """
        self.discovered = False
        self.visited = False
        self.distance = float("inf")
        self.item = item
        self.index = None
        self.edges = [] # adjacency list
        self.previous = None # for backtracking
        self.has_passenger = False
        self.has_passenger_along = False # to decide if second weight should be used


    def __str__(self):
        """
        Returns a string representation of the vertex which includes its distance and position in the heap
        
        """

        if self.has_passenger:
            _string = "Special "
        else:
            _string = ""
        return_str = _string + str(self.item) + " with distance " + str(self.distance) + " and position in heap: " + str(self.index)
        return return_str


class Edge:
    """
    Edge class. Stores the destination vertex and the two weights

    """
    def __init__(self, v, w1, w2):
        """
        Initialize the vertex. Set distance to infinity
        
        :Input:
            item: Integer representing the vertex. 
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        
        """
        self.v = v
        self.actual_weight = None
        self.w1 = w1
        self.w2 = w2

    def __str__(self):
        return_string = ""
        return_string += f"({self.v.item}, {self.w1}, {self.w2})"
        return return_string

def setup_mapping(num_vertices, passengers, roads):
    """
    Helper function which processes the input edges and returns a list of vertices, each vertex has its list of edges.
    For example, Vertex 3 is at position 3 in this returned list and can be accessed via list[3]

    :Input:
        num_vertices: number of vertices
        passengers: the vertices with passengers
        roads: the roads(edges)
    :Return: List of vertices
    :Time complexity: O(V + E)
    :Aux space complexity: O(V + E + P), where V is the number of vertices and E is the number of edges and P is the number of passengers
    """

    vertex_with_edges = [None for _ in range(num_vertices)]  # O(V) auxiliary space

    # O(E) time complexity, O(E) auxiliary space because the sum of edge objects created for all vertices is E
    # Create a vertex at the index equal to the vertex itself if that location does not contain a created Vertex yet, and add edge from vertex1 to vertex2
    for i in range(len(roads)):
        if vertex_with_edges[roads[i][0]] is None:
            start_vertex = Vertex(roads[i][0])
            start_vertex.index = roads[i][0] # set index of vertex, useful for inserting into heap to avoid NoneType error
            vertex_with_edges[roads[i][0]] = start_vertex
        if vertex_with_edges[roads[i][1]] is None:
            end_vertex = Vertex(roads[i][1])
            end_vertex.index = roads[i][1] # set index of vertex, useful for inserting into heap to avoid NoneType error
            vertex_with_edges[roads[i][1]] = end_vertex

        # first vertex(roads[i][0]) has a new edge to second item(roads[i][1])
        vertex_with_edges[roads[i][0]].edges.append(Edge(vertex_with_edges[roads[i][1]], roads[i][2], roads[i][3]))


    # initialize passenger vertices, worst case all locations have passengers , O(P) time complexity
    for location in passengers:
        vertex = vertex_with_edges[location]
        vertex.has_passenger = True
        vertex.has_passenger_along = True

    # for vertex in vertex_with_edges:
    #     print(vertex)


    return vertex_with_edges


def dijkstra(graph, start, end, last_vertex, actual_end):
    """
    Returns a list of integers representing the path with the minimum distance

    :Input:
        graph: The graph data structure
        start: Source vertex item
        end: Destination vertex item
        last_vertex: The last vertex item in the graph
        actual_end: The original destination vertex item
    :Time complexity: Overall O(E log V)
                      From O(V log V + V^2 log V + E), V^2 log V dominates O(V log V) and then O(V^2 log V) is rewritten as O(E log V). This is because when perform edge relaxation, 
                      E log V is the tightest possible worst-case time complexity, as it is more accurate than O(V^2 log V) for sparse graphs. 
                      So we have O(E log V + E) -> O(E (log V+1)) -> O(E log V)
    :Aux space complexity: O(V) for adding V vertices to the heap
    """
    potential_end = None

    def convert_vertex(item):
        """
        Helper function to convert back any linked graph vertices to its original vertex, e.g. vertex 21 is actually the original vertex 1 + number_vertices(20) so convert 21
        back to 1 by doing 21 - 20. 

        :Input:
            item: Vertex item to be converted
        :Return: Converted item, should be in the range(0..last_vertex)
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        if item > last_vertex:
            item = item - (last_vertex + 1)
        return item
    
    heap = MinHeap(graph.num_vertices)
    # initialize heap, O(V log V) time complexity, O(V) aux space complexity for the heap as we are adding V vertices into the heap's array
    
    # To account for invalid inputs, where there might be some vertices in the range(0...L-1) is not connected to the graph
    # remove Nones from heap, and update the indexes to account for removed "Nones", essentially "shuffle" elements to the left. 
    # Eventually left with Nones at the end of the heap array.
    number_of_nones = 0
    for i in range(graph.num_vertices): # O(V)
        if graph.vertices[i] is None:
            number_of_nones += 1
            continue
        if graph.vertices[i] is not None:
            vertex = graph.vertices[i ]
            vertex.index -= number_of_nones
            if i == start:
                vertex.distance = 0
            heap.add(vertex)

    # remove Nones from end of heap array, O(V)
    last = heap.array[-1]
    while last is None:
        heap.array.pop()
        last = heap.array[-1]


    # Overall O(V^2 log V), because examining the outgoing edges of the served vertex and performing heap update is O(V log V) and this is done V times
    minimum = None
    while len(heap.array) > 0: # O(V)
        minimum = heap.getMin()  # get minimum vertex, O(log L)
        # print(f"{minimum} is served from heap. ")

        if minimum.item == end:
            # print(f"minimum is reached {minimum.item}")
            break
        elif minimum.item == actual_end:
            potential_end = minimum
        # heap.print_heap()
        minimum.discovered = True
        minimum.visited = True

        # look at outgoing edges from vertex, and update if necessary, O(V log V)
        for edge in minimum.edges:  # O(L)

            # select edge weight to be used, and set outgoing edges to have_passenger_along so that in future the second weight will be used
            if minimum.has_passenger_along:
                if edge.v.item != start:
                    # print(f"The current vertex is {minimum.item} and the adjacent edge is {edge.v.item}")
                    edge.v.has_passenger_along = True
                    edge.actual_weight = edge.w2
            else:
                edge.actual_weight = edge.w1

            adjacent_vertex = edge.v

            # if its adjacent vertex has not been discovered, update distance from infinity to distance
            if not adjacent_vertex.discovered:  
                # print(f"The current vertex is {minimum.item} and the adjacent edge is {edge.v.item}")
                # print(f"{adjacent_vertex} is discovered so its infinity distance is overwritten. It is at position {adjacent_vertex.index}")
                adjacent_vertex.discovered = True
                adjacent_vertex.distance = minimum.distance + edge.actual_weight
                adjacent_vertex.previous = minimum  # set its previous vertex

                # update heap, O(log V)
                heap.rise(adjacent_vertex.index)
                # heap.print_heap()Fsetup

            # if vertex was discovered before, update distance if smaller distance available
            elif not adjacent_vertex.visited:
                # print(f"{adjacent_vertex} has been discovered but now we need to check if distance is updated")
                if adjacent_vertex.distance > minimum.distance + edge.actual_weight:
                    # print(f"Distance is updated from {adjacent_vertex.distance} to {minimum.distance + edge.actual_weight}")
                    adjacent_vertex.distance = minimum.distance + edge.actual_weight
                    adjacent_vertex.previous = minimum

                    # update heap, O(log V)
                    heap.rise(adjacent_vertex.index)
                    # heap.print_heap()

    # print(f"The end's distance is {minimum.distance}")
    

    # Maybe there was no need to take detour to find passenger. If so, return this shortest path
    route = []
    if potential_end is not None and potential_end.distance < minimum.distance:
        previous = potential_end
    else:
        previous = minimum
    graph.cost = previous.distance
    # print(previous)
        
    # O(E), trace back the path
    while previous is not None:
        route.append(convert_vertex(previous.item))
        if previous.previous is not None and previous.previous.previous is not None:
            if convert_vertex(previous.item) == convert_vertex(previous.previous.item):
                previous = previous.previous.previous
            else:
                previous = previous.previous
        else:
            previous = previous.previous
    route.reverse() # O(E)
    return route


def optimalRoute(start, end, passengers, roads):

    """
    Returns the path with minimum driving distance from start to end.

    Approach:
    Modeled as a graph, the roads are edges, passengers are special vertices. Use dijkstra to find the shortest
    path from start vertex to end vertex. There are two options, either find shortest path from start to end without taking any detours, or
    choose to take one detour to visit a passenger. 

    First process roads(edges) to get the number of vertices, which allows us to get the vertices of the graph as [0... number_of_vertices - 1]

    Once a passenger location(vertex) is visited, all subsequent edges(roads) will use second weight(carpool lane).

    Run dijkstra on a graph with double the required vertices/edges. To differentiate the locations of the two graphs, graph2 vertex = graph1 vertex + number of vertices. 
    e.g. For a graph of 5 vertices, graph1's vertex 0 will have its linked graph vertex as 0 + 5 = 5
    For each passenger location in the passengers argument, add an edge from graph1 vertex to its corresponding graph2 vertex. This links the two graphs.
    This prevents us from overlooking the fact that taking a detour, while initially costing more at first, may pay off in the long run.

    Because of the linking, when dijkstra is run, the shortest path will be found from the source to the passenger, then from the passenger to the
    destination. If multiple detours to pick up passengers are available, only the best detour will be chosen.

    Compare the two results(detour vs no detour) to decide if making the detour is worth it, and return the optimum path.

    :Precondition: 
        for each tuple(a,b,c,d) in roads, d<=c
        for each integer in {start..end}, the integer appears at least once as either a or b in tuple(a, b, c, d)
        there is a path from start to end & start != end
        start not in passengers and end not in passengers
    :Input:
        start:Source
        end:Destination
        passengers:Locations that contain a passenger that can be picked up
        roads:Roads than can be used, they have two lanes with two driving times

    :Return: Optimal route in the form of a list of ordered locations, e.g. [0, 1, 6, 4]
    :Time complexity: O(E(log V) + V + P) where E is the number of roads(edges), V is the number of locations(vertices) P is the number of passengers, which is within the requirement of O(E log V).
                      Since our graph is connected, E >= V, and given that P <= V <= E, E(log V) is greater than V and P, so we have O(E log V) overall
    :Aux space complexity: O(E + P + V), given that the passengers are a subset of the vertices, we know P <= V, V is greater than P, so we have overall O(E + V)

    """

    # to get the number of locations(vertices), O(E) assuming integer comparisons are O(1)
    last_vertex = 0
    for road in roads:  #
        larger_location = max(road[0], road[1])  # O(1)
        if larger_location > last_vertex:  # O(1)
            last_vertex = larger_location
    num_locations = last_vertex + 1

    # create a new copy of the edges, also change the vertex values, e.g. if graph has 12 vertices from 0 to 11 , the new vertices are [12..23], O(2E) time and aux space 
    additional_roads1 = [None for i in range(len(roads))] 
    additional_roads2 = [None for i in range(len(roads))]
    for i in range(len(roads)):
        additional_roads1[i] = (roads[i][0], roads[i][1], roads[i][2], roads[i][3])
        additional_roads2[i] = (roads[i][0]+num_locations, roads[i][1]+num_locations, roads[i][2], roads[i][3])
    additional_roads1.extend(additional_roads2)

    # add the linking edge for each passenger, this is what "links" the two graphs, O(P) time
    for passenger in passengers:
        additional_roads1.append((passenger, passenger + num_locations, 0, 0)) # O(P) aux space because add more edges to the roads

    # O(2V+2E+2P) time and aux space complexity
    newMapping = setup_mapping(num_locations*2, passengers, additional_roads1)


    graph = Graph(newMapping) # O(1)3

    # 2E log 2V time complexity, 2V aux space complexity for the heap, the constant 2 can be ignored
    route = dijkstra(graph, start, end+num_locations, last_vertex, end)

    return route
    

