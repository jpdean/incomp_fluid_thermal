import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.mesh import create_cell_partitioner, GhostMode


def generate(comm, h=0.1, h_fac=1/3):
    gmsh.initialize()

    volume_id = {"fluid": 1}

    boundary_id = {"inlet": 2,
                   "outlet": 3,
                   "wall": 4,
                   "obstacle": 5}
    gdim = 2

    if comm.rank == 0:

        gmsh.model.add("model")
        factory = gmsh.model.geo

        length = 2.2
        height = 0.41
        c = (0.2, 0.2)
        r = 0.05

        rectangle_points = [
            factory.addPoint(0.0, 0.0, 0.0, h),
            factory.addPoint(length, 0.0, 0.0, h),
            factory.addPoint(length, height, 0.0, h),
            factory.addPoint(0.0, height, 0.0, h)
        ]

        circle_points = [
            factory.addPoint(c[0], c[1], 0.0, h),
            factory.addPoint(c[0] + r, c[1], 0.0, h * h_fac),
            factory.addPoint(c[0], c[1] + r, 0.0, h * h_fac),
            factory.addPoint(c[0] - r, c[1], 0.0, h * h_fac),
            factory.addPoint(c[0], c[1] - r, 0.0, h * h_fac)
        ]

        rectangle_lines = [
            factory.addLine(rectangle_points[0], rectangle_points[1]),
            factory.addLine(rectangle_points[1], rectangle_points[2]),
            factory.addLine(rectangle_points[2], rectangle_points[3]),
            factory.addLine(rectangle_points[3], rectangle_points[0])
        ]

        circle_lines = [
            factory.addCircleArc(
                circle_points[1], circle_points[0], circle_points[2]),
            factory.addCircleArc(
                circle_points[2], circle_points[0], circle_points[3]),
            factory.addCircleArc(
                circle_points[3], circle_points[0], circle_points[4]),
            factory.addCircleArc(
                circle_points[4], circle_points[0], circle_points[1])
        ]

        rectangle_curve = factory.addCurveLoop(rectangle_lines)
        circle_curve = factory.addCurveLoop(circle_lines)

        square_surface = factory.addPlaneSurface(
            [rectangle_curve, circle_curve])
        # circle_surface = factory.addPlaneSurface([circle_curve])

        factory.synchronize()

        gmsh.model.addPhysicalGroup(2, [square_surface], volume_id["fluid"])
        # gmsh.model.addPhysicalGroup(2, [circle_surface], omega_1)

        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[0], rectangle_lines[2]],
            boundary_id["wall"])
        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[1]], boundary_id["outlet"])
        gmsh.model.addPhysicalGroup(
            1, [rectangle_lines[3]], boundary_id["inlet"])
        gmsh.model.addPhysicalGroup(1, circle_lines, boundary_id["obstacle"])

        gmsh.model.mesh.generate(2)

        # gmsh.fltk.run()

    partitioner = create_cell_partitioner(GhostMode.shared_facet)
    msh, _, ft = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=gdim, partitioner=partitioner)
    ft.name = "Facet markers"

    return msh, ft, boundary_id


if __name__ == "__main__":
    msh, ft, boundary_id = generate(MPI.COMM_WORLD, h=0.05)

    from dolfinx import io
    with io.XDMFFile(msh.comm, "benchmark_mesh.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_meshtags(ft)
