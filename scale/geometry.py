class SPHGeom:
    def __init__(self):
        self.source = None
        self.boundaries = None
        self.sink = None


class USDAGeom(SPHGeom):
    pass


class CubeGeom(SPHGeom):
    def __init__(self, DOMAIN_WIDTH, DOMAIN_HEIGHT):
        domain_x_lim = np.array([
            SMOOTHING_LENGTH,
            geometry.boundary.xmax - SMOOTHING_LENGTH,
        ])
        domain_y_lim = np.array([
            SMOOTHING_LENGTH,
            geometry.ymax - SMOOTHING_LENGTH,
        ])
