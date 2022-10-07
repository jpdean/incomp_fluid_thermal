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

    volume_id = {"fluid": 1}

    boundary_id = {"inlet": 2,
                   "outlet": 3,
                   "wall": 4,
                   "obstacle": 5}

    res_min = r / 3

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        obstacle_tags = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle_tags)])
        gmsh.model.occ.synchronize()

        # Get volumes in model (as list of (dim, tag))
        volumes = gmsh.model.getEntities(dim=gdim)
        assert (len(volumes) == 1)  # Just one volume for fluid
        # Tag fluid
        gmsh.model.addPhysicalGroup(
            volumes[0][0], [volumes[0][1]], volume_id["fluid"])
        gmsh.model.setPhysicalName(volumes[0][0], volume_id["fluid"], "Fluid")

        inflow_tags, outflow_tags, walls_tags, obstacle_tags = [], [], [], []

        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(
                boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H/2, 0]):
                inflow_tags.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H/2, 0]):
                outflow_tags.append(boundary[1])
            elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
                walls_tags.append(boundary[1])
            else:
                obstacle_tags.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls_tags, boundary_id["wall"])
        gmsh.model.setPhysicalName(1, boundary_id["wall"], "Walls")
        gmsh.model.addPhysicalGroup(1, inflow_tags, boundary_id["inlet"])
        gmsh.model.setPhysicalName(1, boundary_id["inlet"], "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow_tags, boundary_id["outlet"])
        gmsh.model.setPhysicalName(1, boundary_id["outlet"], "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle_tags, boundary_id["obstacle"])
        gmsh.model.setPhysicalName(1, boundary_id["obstacle"], "Obstacle")

        # Create distance field from obstacle.
        # Add threshold of mesh sizes based on the distance field
        # LcMax -                  /--------
        #                      /
        # LcMin -o---------/
        #        |         |       |
        #       Point    DistMin DistMax
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle_tags)
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


if __name__ == "__main__":
    msh, ft, boundary_id = generate()

    from dolfinx import io
    with io.XDMFFile(msh.comm, "benchmark_mesh.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_meshtags(ft)
