# From Jørgen's FEniCSx tutorial. TODO Simplify

import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.mesh import create_cell_partitioner, GhostMode


def generate():
    gmsh.initialize()

    L = 2.2
    H = 0.41
    c_x = c_y = 0.2
    r = 0.05
    gdim = 2
    order = 1

    boundary_id = {"inlet": 2,
                   "outlet": 3,
                   "wall": 4,
                   "obstacle": 5}

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

    if mesh_comm.rank == model_rank:
        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
        gmsh.model.occ.synchronize()

    fluid_marker = 1
    if mesh_comm.rank == model_rank:
        volumes = gmsh.model.getEntities(dim=gdim)
        assert (len(volumes) == 1)
        gmsh.model.addPhysicalGroup(
            volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

    inflow, outflow, walls, obstacle = [], [], [], []
    if mesh_comm.rank == model_rank:
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(
                boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H/2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H/2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, boundary_id["wall"])
        gmsh.model.setPhysicalName(1, boundary_id["wall"], "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, boundary_id["inlet"])
        gmsh.model.setPhysicalName(1, boundary_id["inlet"], "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, boundary_id["outlet"])
        gmsh.model.setPhysicalName(1, boundary_id["outlet"], "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, boundary_id["obstacle"])
        gmsh.model.setPhysicalName(1, boundary_id["obstacle"], "Obstacle")

    # Create distance field from obstacle.
    # Add threshold of mesh sizes based on the distance field
    # LcMax -                  /--------
    #                      /
    # LcMin -o---------/
    #        |         |       |
    #       Point    DistMin DistMax
    res_min = r / 3
    if mesh_comm.rank == model_rank:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(
            threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.2 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    if mesh_comm.rank == model_rank:
        # gmsh.option.setNumber("Mesh.Algorithm", 8)
        # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.optimize("Netgen")

    partitioner = create_cell_partitioner(GhostMode.shared_facet)
    msh, _, ft = gmshio.model_to_mesh(
        gmsh.model, mesh_comm, model_rank, gdim=gdim, partitioner=partitioner)
    ft.name = "Facet markers"

    return msh, ft, boundary_id
