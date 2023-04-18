from Graph import *

def setup_mapping(num_vertices, passengers, roads):
    num_locations = num_vertices

    # initialize vertex and edges for it
    vertex_with_edges = [None for _ in range(num_locations)]  # O(|L|)
    # O(|R|)
    # for road in roads:
    for i in range(len(roads)):
        if vertex_with_edges[roads[i][0]] is None:
            start_vertex = Vertex(roads[i][0])
            start_vertex.index = roads[i][0]
            vertex_with_edges[roads[i][0]] = start_vertex
        if vertex_with_edges[roads[i][1]] is None:
            end_vertex = Vertex(roads[i][1])
            end_vertex.index = roads[i][1]
            vertex_with_edges[roads[i][1]] = end_vertex
        vertex_with_edges[roads[i][0]].edges.append(Edge(vertex_with_edges[roads[i][1]], roads[i][2], roads[i][3]))


    # initialize passenger vertices, O(L)
    for location in passengers:
        vertex = vertex_with_edges[location]
        vertex.has_passenger = True
        vertex.has_passenger_along = True

    return vertex_with_edges


def dijkstra(graph) -> list:
    """
    :complexity: O(L^2 log L)-> O(R log L)

    """
    # O(L)
    minimum = None
    while len(graph.heap.array) > 0:
        minimum = graph.heap.getMin()  # get minimum vertex, O(log L)
        print(f"{minimum} is served from heap. ")

        if minimum.item == graph.end:
            print(f"minimum is reached {minimum.item}")
            break
        graph.heap.print_heap()
        minimum.discovered = True
        minimum.visited = True

        # look at outgoing edges from source, O(L log L)
        for edge in minimum.edges:  # O(L)

            if minimum.has_passenger_along:
                if edge.v.item != graph.start:
                    print(f"The current vertex is {minimum.item} and the adjacent edge is {edge.v.item}")
                    print(minimum.has_passenger_along)
                    edge.v.has_passenger_along = True
                    edge.actual_weight = edge.w2

            else:
                edge.actual_weight = edge.w1

            adjacent_vertex = edge.v

            if not adjacent_vertex.discovered:  # if its adjacent vertex has not been discovered, add distance
                print(f"The current vertex is {minimum.item} and the adjacent edge is {edge.v.item}")
                print(f"{adjacent_vertex} is discovered so its infinity distance is overwritten. It is at position {adjacent_vertex.index}")
                adjacent_vertex.discovered = True
                adjacent_vertex.distance = minimum.distance + edge.actual_weight
                adjacent_vertex.previous = minimum  # set its previous vertex

                # update heap, O(log L)
                graph.heap.rise(adjacent_vertex.index)
                graph.heap.print_heap()

            elif not adjacent_vertex.visited:
                print(f"{adjacent_vertex} has been discovered but now we need to check if distance is updated")
                if adjacent_vertex.distance > minimum.distance + edge.actual_weight:
                    print(f"Distance is updated from {adjacent_vertex.distance} to {minimum.distance + edge.actual_weight}")
                    adjacent_vertex.distance = minimum.distance + edge.actual_weight
                    adjacent_vertex.previous = minimum

                    # update heap, O(log L)
                    graph.heap.rise(adjacent_vertex.index)
                    graph.heap.print_heap()

    last_vertex = graph.last_vertex
    print(last_vertex)
    if minimum.item > last_vertex:
        route = [minimum.item - graph.num_vertices//2]
    else:
        route = [minimum.item]
    previous = minimum.previous
    graph.cost = minimum.distance

    while previous is not None:
        if previous.item > last_vertex:
            vertex_item = previous.item - graph.num_vertices//2
        else:
            vertex_item = previous.item
        route.append(vertex_item)

        # removes duplicate
        prev2_item = None
        if previous.previous is not None and previous.previous.previous is not None:
            if previous.previous.item > last_vertex:
                prev2_item = previous.previous.item - graph.num_vertices//2
            else:
                prev2_item = previous.previous.item
        if vertex_item == prev2_item:
            previous = previous.previous.previous
        else:
            previous = previous.previous

    route.reverse()  # O(|R|)
    return route


def optimalRoute(start, end, passengers, roads):

    """
    Returns the path with minimum driving distance from start to end.

    Approach:
    Modeled as a graph, the roads are edges, passengers are special vertices. Use dijkstra to find the shortest
    path from start vertex to end vertex.

    First process roads(edges) to get the number of vertices, which allows us to get the vertices of the graph as
    [0... number_of_vertices - 1]

    Run dijkstra on a normal graph from start to end to find minimum driving distance. If passenger locations are
    on the way, the algorithm ensures that all roads traversed after a passenger is picked up will take the carpool lane.

    Run dijkstra on a graph with double the vertices of the first graph. To differentiate the locations of the two graphs,
    graph2 vertex = graph1 vertex + (number of vertices). e.g. vertex  For each passenger location in the
    passengers argument, add an edge from graph1 vertex to its corresponding graph2 vertex. This links the two graphs.
    So when dijkstra is run, the shortest path will be found from the source to the passenger, then from the passenger to the
    destination. If multiple detours to pick up passengers are available, only the best detour will be chosen.

    Finally, compare the two results to decide if making the detour is worth it, and return the optimum path

    Parameters:

    start:Source
    end:Destination
    passengers:Locations that contain a passenger that can be picked up
    roads:Roads than can be used, they have two lanes with two driving times

    :return: Optimal route in the form of a list of ordered locations, e.g. [0, 1, 6, 4]
    """

    # to get the number of locations(vertices), O(|R|) assuming integer comparisons are O(1)
    last_vertex = 0
    for road in roads:  # O(|R|)
        larger_location = max(road[0], road[1])  # O(1)
        if larger_location > last_vertex:  # O(1)
            last_vertex = larger_location
    num_locations = last_vertex + 1
    print(num_locations)

    # run a normal dijkstra(no linking)
    mapping = setup_mapping(num_locations, passengers, roads)
    graph = Graph(mapping, start, end, last_vertex)
    route1 = dijkstra(graph)

    # double the vertices, also change the vertex values, e.g. 0 is 12, 1 is 13... 11 is 23
    additional_roads = [None for i in range(len(roads))]
    for i in range(len(additional_roads)):
        additional_roads[i] = (roads[i][0]+num_locations, roads[i][1]+num_locations, roads[i][2], roads[i][3])
    roads.extend(additional_roads)

    # add the linking edge for each passenger
    for passenger in passengers:
        roads.append((passenger, passenger + num_locations, 0, 0))

    newMapping = setup_mapping(num_locations*2, passengers, roads)

    graph2 = Graph(newMapping, start, end+num_locations, last_vertex)

    route2 = dijkstra(graph2)

    print(f"Non-linking graph: {graph.cost}, linking graph: {graph2.cost}")
    if graph.cost > graph2.cost:
        return route2
    else:
        return route1


if __name__ == '__main__':
    # given example, taking detour helps shorten
    print(optimalRoute(0, 4, [2, 1],
                       [(0, 3, 5, 3), (3, 4, 35, 15), (3, 2, 2, 2), (4, 0, 15, 10), (2, 4, 30, 25), (2, 0, 2, 2),
                        (0, 1, 10, 10), (1, 4, 30, 20)]))

    # some passengers are out of the way and they cannot reach destination
    # print(optimalRoute(0, 4, [3, 5, 11],
    #                    [(1,4, 1, 1),(0, 1, 1, 1), (1, 2, 1, 1), (1, 6, 10, 1), (2, 3, 1, 1), (3, 7, 1, 1), (0, 5, 1, 1),(7,8,1,1), (8,1,1,1),(6,4,42,20),(1,9,1,1),(9,11,1,1),(11,10,1,1),(10,1,1,1)]))

    # no passengers
    # print(optimalRoute(0, 3, [],
    #                    [(0, 1, 1, 1), (0, 2, 6, 6), (0, 4, 11, 11), (1, 3, 25, 15), (2, 3, 25, 8), (4, 3, 25, 1)]))

    # some passengers are out of the way and they cannot reach destination, more complicated
    # print(optimalRoute(0, 4, [3, 5, 11], [(0, 1, 1, 1), (1, 2, 1, 1), (1, 6, 1, 1), (2, 3, 1, 1), (3, 7, 1, 1), (0, 5, 1, 1), (7, 8, 1, 1), (8, 1, 1, 1), (6, 4, 4, 1), (1, 9, 1, 1), (9, 11, 1, 1), (11, 10, 1, 1), (10, 1, 1, 1)]))

    # # some passengers are out of the way and they cannot reach destination, more complicated
    # print(optimalRoute(0, 4, [3, 5, 11], [(0, 1, 1, 1), (1, 2, 1, 1), (1, 6, 1, 1), (2, 3, 1, 1), (3, 7, 1, 1), (0, 5, 1, 1), (7, 8, 1, 1), (8, 1, 1, 1), (6, 4, 6, 1), (1, 9, 1, 1), (9, 11, 1, 1), (11, 10, 1, 1), (10, 1, 1, 1)]))

