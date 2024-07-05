def distance_of_points(p1, p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    d = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
