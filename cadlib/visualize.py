import multiprocessing
from pathlib import Path
from OCC.Core.Geom import Geom_Curve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.TopoDS import TopoDS_Iterator
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Circ, gp_Pln, gp_Vec, gp_Ax3, gp_Ax2, gp_Ax1, gp_Lin
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeVertex,BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Copy)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet,BRepFilletAPI_MakeChamfer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Extend.DataExchange import write_stl_file
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone
from copy import copy
from .extrude import *
from .sketch import Loop, Profile
from .curves import *
import os
import trimesh
from trimesh.sample import sample_surface
import random
#from utils.Viewer import display

def vec2CADsolid(vec, scale=None, is_numerical=True, n=256):
    cad = CADSequence.from_vector(vec, is_numerical=False, n=256)
    # cad = CADSequence.from_vector(vec, is_numerical=False, n=256)
    # print(cad)
    if scale is not None:
        cad.transform(0.0, scale)
    cad.denumericalize()
    cad = create_CAD(cad)
    return cad

# def create_CAD(cad_seq: CADSequence):
#     """create a 3D CAD model from CADSequence. Only support extrude with boolean operation."""
#     command = cad_seq.seq[0]
#     if isinstance(command, Extrude):
#         body = create_by_extrude(command)
#     elif isinstance(command, Revolve):
#         body = create_by_revolve(command)
#     else:
#         raise Exception('Feature is not support now')
#     fillet_list = []
#     chamfer_list = []

#     for command in cad_seq.seq[1:]:
#         # if not isinstance(command, Fillet) and len(fillet_list)>0:
#         #     body = create_by_fillet(body, fillet_list)
#         #     fillet_list.clear()
#         # if not isinstance(command, Chamfer) and len(chamfer_list)>0:
#         #     create_by_chamfer(body, chamfer_list)
#         #     chamfer_list.clear()
#         if isinstance(command, Extrude):
#             new_body = create_by_extrude(command)
#             if(new_body is not None):
#                 body_backup = BRepBuilderAPI_Copy(body).Shape()
#                 if command.type == 0:
#                     fuse = BRepAlgoAPI_Fuse(body, new_body)
#                     if fuse.IsDone():
#                         body = fuse.Shape()
#                     else:
#                         body = body_backup
#                 elif command.type == 1:
#                     cut = BRepAlgoAPI_Cut(body, new_body)
#                     if cut.IsDone():
#                         body = cut.Shape()
#                     else:
#                         body = body_backup
#             else:
#                 continue
#         elif isinstance(command, Revolve):
#             new_body = create_by_revolve(command)
#             if(new_body is not None):
#                 body_backup = BRepBuilderAPI_Copy(body).Shape()
#                 if command.type == 0:
#                     fuse = BRepAlgoAPI_Fuse(body, new_body)
#                     if fuse.IsDone():
#                         body = fuse.Shape()
#                     else:
#                         body = body_backup
#                 elif command.type == 1:
#                     cut = BRepAlgoAPI_Cut(body, new_body)
#                     if cut.IsDone():
#                         body = cut.Shape()
#                     else:
#                         body = body_backup
#             else:
#                 continue
#         elif isinstance(command, Fillet):
#             fillet_list.append(command)
#         elif isinstance(command, Chamfer):
#             chamfer_list.append(command)
#     if(len(fillet_list)>0):
#         body = create_by_fillet(body, fillet_list)
#     if(len(chamfer_list)>0):
#         body = create_by_chamfer(body, chamfer_list)
#     return body

def one_op_one_body(cad_op):
    if isinstance(cad_op, Extrude):
        assert cad_op.type == 0, "Only ExtrusionBoss is allowed."
        new_body = create_by_extrude(cad_op)
        # assert new_body, "Extrude failed!"
        if new_body == None:
            print("Extrude failed!")
            return None
        return new_body
    elif isinstance(cad_op, Fillet):
        body = create_by_fillet(cad_op, body)
        if body == None:
            print("Fillet failed!")
            return None
        return body
    elif isinstance(cad_op, Chamfer):
        body = create_by_chamfer(cad_op, body)
        if body == None:
            print("Chamfer failed!")
            return None
        return body
    elif isinstance(cad_op, Revolve):
        assert cad_op.type == 0, "Only Revolution is allowed."
        new_body = create_by_revolve(cad_op)
        # assert new_body, "Revolute failed!"
        if new_body == None:
            print("Revolute failed!")
            return None
        return new_body
    else:
        return None

def process_op(body, cad_op):
    if isinstance(cad_op, Extrude):
        new_body = create_by_extrude(cad_op)
        # assert new_body, "Extrude failed!"
        if new_body == None:
            print("Extrude failed!")
            return body
        if cad_op.type == 0 and body is not None:
            body = BRepAlgoAPI_Fuse(body, new_body).Shape()
        elif cad_op.type == 1 and body is not None:
            body = BRepAlgoAPI_Cut(body, new_body).Shape()
    elif isinstance(cad_op, Fillet):
        body = create_by_fillet(cad_op, body)
    elif isinstance(cad_op, Chamfer):
        body = create_by_chamfer(cad_op, body)
    elif isinstance(cad_op, Revolve):
        new_body = create_by_revolve(cad_op)
        # assert new_body, "Revolute failed!"
        if new_body == None:
            print("Revolute failed!")
            return body
        if cad_op.type == 0 and body is not None:
            body = BRepAlgoAPI_Fuse(body, new_body).Shape()
        elif cad_op.type == 1 and body is not None:
            body = BRepAlgoAPI_Cut(body, new_body).Shape()
    return body

def create_CAD(cad_seq: CADSequence, use_reverse=False):
    """create a 3D CAD model from CADSequence. Only support extrude with boolean operation."""
    count = 0
    body = None

    for cad_op in cad_seq.seq:
        new_body = None

        #have reverse operation
        if isinstance(cad_op, Extrude):
            new_body = create_by_extrude(cad_op)
            if new_body == None:
                print("Extrude failed!")
                continue
            if cad_op.type == 0 and body is not None:
                body = BRepAlgoAPI_Fuse(body, new_body).Shape()
            elif cad_op.type == 1 and body is not None:
                body = BRepAlgoAPI_Cut(body, new_body).Shape()
        elif isinstance(cad_op, Fillet):
            body = create_by_fillet(cad_op, body)
        elif isinstance(cad_op, Chamfer):
            body = create_by_chamfer(cad_op, body)
        elif isinstance(cad_op, Revolve):
            new_body = create_by_revolve(cad_op)
            if new_body == None:
                print("Revolute failed!")
                continue
            if cad_op.type == 0 and body is not None:
                body = BRepAlgoAPI_Fuse(body, new_body).Shape()
            elif cad_op.type == 1 and body is not None:
                body = BRepAlgoAPI_Cut(body, new_body).Shape()

        count += 1

        if body is None:
            body = new_body

    return body

def create_by_extrude(extrude_op: Extrude):
    """create a solid body from Extrude instance."""
    profile = copy(extrude_op.profile) # use copy to prevent changing extrude_op internally
    profile.denormalize(extrude_op.sketch_size, size=256)
    sketch_plane = copy(extrude_op.sketch_plane)
    sketch_plane.origin = extrude_op.sketch_pos
    face = create_profile_face(profile, sketch_plane)
    normal = gp_Dir(*extrude_op.sketch_plane.normal)
    if extrude_op.type == 1:
        normal = normal.Reversed()
    # if  extrude_op.isreversed:
    #     normal = normal.Reversed()
    body = None
    if(extrude_op.extent_one < 1e-5 and extrude_op.extent_two < 1e-5):
        return body
    if(extrude_op.extent_one > 1e-5):
        ext_vec = gp_Vec(normal).Multiplied(extrude_op.extent_one*1.)
        prism = BRepPrimAPI_MakePrism(face, ext_vec)
        if prism.IsDone():
            body = prism.Shape()
        else:
            return None
    # if extrude_op.extent_type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
    #     body_sym = BRepPrimAPI_MakePrism(face, ext_vec.Reversed()).Shape()
    #     body = BRepAlgoAPI_Fuse(body, body_sym).Shape()
    # if extrude_op.extent_type == EXTENT_TYPE.index("BothSidesFeatureExtentType"):
    if(extrude_op.extent_two > 1e-5):
        ext_vec = gp_Vec(normal.Reversed()).Multiplied(extrude_op.extent_two*1.)
        prism_two = BRepPrimAPI_MakePrism(face, ext_vec)
        if prism_two.IsDone():
            body_two = prism_two.Shape()
        else:
            body_two = None
        if(extrude_op.extent_one > 1e-5 and body is not None and body_two is not None):
            fuse = BRepAlgoAPI_Fuse(body, body_two)
            if fuse.IsDone():
                body = fuse.Shape()
            # else keep body as is
        elif body_two is not None:
            body = body_two
    return body

def create_by_revolve(revolve_op: Revolve):
    profile = copy(revolve_op.profile)  # use copy to prevent changing extrude_op internally
    profile.denormalize(revolve_op.sketch_size, size=256)
    sketch_plane = copy(revolve_op.sketch_plane)
    sketch_plane.origin = revolve_op.sketch_pos
    # print("sketch plane: ", sketch_plane)
    face = create_profile_face(profile, sketch_plane)
    # display(face)
    axis = revolve_op.axis
    # print(axis.cent)
    # print(axis.direction)
    # print(axis.sketch_plane)
    # print(sketch_plane.origin)
    # print(point_local2global(axis.cent, sketch_plane, to_gp_Pnt=False))
    # print(point_local2global(axis.direction, sketch_plane, direction=True, to_gp_Pnt=False))

    revolve_axis = gp_Ax1(point_local2global(axis.cent, axis.sketch_plane),gp_Dir(*point_local2global(axis.direction, axis.sketch_plane, direction=True, to_gp_Pnt=False)))    # direction=True before
    angle = revolve_op.angle
    revol = BRepPrimAPI_MakeRevol(face, revolve_axis, angle*1.)
    if revol.IsDone():
        body = revol.Shape()
    else:
        body = None
    #display(body)
    return body

# def create_by_fillet(solid_model, fillet_op_list: list):
#     # fillet = BRepFilletAPI_MakeFillet(solid_model)
#     d_e = []
#     for fillet_op in fillet_op_list:
#         d1 = fillet_op.d1 * 1.
#         d2 = fillet_op.d2 * 1.
#         P = gp_Pnt(*fillet_op.cent)
#         # v = BRepBuilderAPI_MakeVertex(P).Vertex()
#         # e = select_edge(solid_model, v)
#         e, _ = seek_edge_by_point(solid_model, P)
#         if e is not None:
#             d_e.append((d1, d2, e))
#     print("d_e length: ", len(d_e))
#     if len(d_e)>0:
#         for d1,d2,e in d_e:
#             # fillet.Add(d1, d2, e)
#             # fillet.Build()
#             # if not fillet.IsDone():
#             #     print("fillet is not done")
#             # if not fillet.IsDone():
#             #     print("fillet is not done")
#             fillet = BRepFilletAPI_MakeFillet(solid_model)
#             fillet.Add(d1, d2, e)
#             fillet.Build()
#             if not fillet.IsDone():
#                 print("fillet is not done")
#                 continue
#             solid_model = fillet.Shape()
#         # fillet.Build()  # If you want to use IsDone, use Build first
#         # if not fillet.IsDone():
#         #     print("fillet is not done")
#         #     return solid_model
#         # return fillet.Shape()
#         return solid_model
#     else:
#         return solid_model

# def create_by_chamfer(solid_model, chamfer_op_list: list):
#     chamfer = BRepFilletAPI_MakeChamfer(solid_model)
#     d_e = []
#     for chamfer_op in chamfer_op_list:
#         d1 = chamfer_op.d1 * 1.
#         d2 = chamfer_op.d2 * 1.
#         # assert d1 == d2, "d1 not equals d2"
#         P = gp_Pnt(*chamfer_op.cent)
#         v = BRepBuilderAPI_MakeVertex(P).Vertex()
#         # e = select_edge(solid_model, v)
#         e, f = seek_edge_by_point(solid_model, P, True)
#         if e is not None:
#             d_e.append((d1, d2, e, f))
#     if(len(d_e)>0):
#         for d1,d2,e,f in d_e:
#             chamfer.Add(d1,d2,e,f)
#         chamfer.Build()
#         if not chamfer.IsDone():
#             print("chamfer is not done")
#             return solid_model
#         return chamfer.Shape()
#     else:
#         return solid_model

# def select_edge(solid_model, Vt):
#     dis = []
#     es = []
#     edges = TopologyExplorer(solid_model).edges()
#     for e in edges:
#         # first, end = BRep_Tool.Range(e)
#         curve_e = BRep_Tool.Curve(e)[0]
#         if not isinstance(curve_e, Geom_Curve):
#             continue
#         p = BRep_Tool.Pnt(Vt)
#         ext = GeomAPI_ProjectPointOnCurve(p, curve_e)
#         if ext.NbPoints()>0:
#             lowdistance = ext.LowerDistance()
#             dis.append(lowdistance)
#             es.append(e)
#     if len(dis) == 0:
#         return None
#     dis = np.array(dis)
#     index = dis.argmin()
#     return es[index]

def create_by_fillet(cad_op : Fillet, body):
    # body_backup = deepcopy(body)
    body_backup = BRepBuilderAPI_Copy(body).Shape()
    try:
        assert body, "No shape defined."
        edge_point = gp_Pnt(cad_op.cent[0], cad_op.cent[1], cad_op.cent[2])
        fillet = BRepFilletAPI_MakeFillet(body)
        edge, _ = seek_edge_by_point(body, edge_point)
        # v = BRepBuilderAPI_MakeVertex(edge_point).Vertex()
        # edge = select_edge(body, v)
        
        # Run the fillet.Add operation with timeout
        success, shape = apply_fillet_with_timeout(fillet, cad_op.d1, cad_op.d2, cad_op.is_sym, edge, timeout=5, use_timeout=False)
        if success:
            return shape
        else:
            print("Fillet.Add operation timed out or Fillet.Build failed. Using original body.")
            return body_backup
    except Exception as e:
        print(e)
        return body_backup

def create_by_chamfer(cad_op: Chamfer, body):
    # body_backup = deepcopy(body)
    body_backup = BRepBuilderAPI_Copy(body).Shape()
    try:
        assert body, "No shape defined."
        edge_point = gp_Pnt(cad_op.cent[0], cad_op.cent[1], cad_op.cent[2])
        chamfer = BRepFilletAPI_MakeChamfer(body)
        edge, face = seek_edge_by_point(body, edge_point, True)
        
        # Run the chamfer.Add operation with timeout
        success, shape = apply_chamfer_with_timeout(chamfer, cad_op.d1, cad_op.d2, edge, timeout=5, face=face, use_timeout=False)
        if success:
            return shape
        else:
            print("Chamfer.Add operation timed out or Chamfer.Build failed. Using original body.")
            return body_backup
    except Exception as e:
        print(e)
        # shape = body
        return body_backup

def seek_edge_by_point(body, edge_point, find_face=False):
    edge_explorer = TopExp_Explorer(body, TopAbs_EDGE)
    target_edge = None
    adjacent_face = None
    min_edge_distance = float('inf')
    
    # max_iterations = 10000  # Safety limit to prevent infinite loops
    # iteration_count = 0
    
    if find_face:
        # Create a map of edges to faces
        edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(body, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    
    while edge_explorer.More(): # and iteration_count < max_iterations:
        # iteration_count += 1
        current = edge_explorer.Current()
        # Check if current is valid before proceeding
        if current.IsNull():
            edge_explorer.Next()
            continue
            
        # Convert to edge
        edge = topods.Edge(current)
        # Skip invalid edges
        if edge.IsNull():
            edge_explorer.Next()
            continue
        
        # Get curve adaptor and check if it's valid
        curve_adaptor = BRepAdaptor_Curve(edge)
        # Get curve with extra checks
        curve_obj = curve_adaptor.Curve()
        curve = curve_obj.Curve()
        if curve is None:
            edge_explorer.Next()
            continue
        
        trsf = curve_adaptor.Trsf()
        curve.Transform(trsf)

        # Attempt to create the projector with safety check
        projector = GeomAPI_ProjectPointOnCurve(edge_point, curve)
        
        if projector.NbPoints() > 0:
            min_distance = projector.LowerDistance()
            if min_edge_distance > min_distance:
                min_edge_distance = min_distance
                target_edge = edge
                
                # Try to find an adjacent face for this edge
                if find_face and edge_face_map.Contains(edge):
                    faces = edge_face_map.FindFromKey(edge)
                    if faces.Size() > 0:
                        adjacent_face = topods.Face(faces.First())
        
        # Transform the edge back to its original position
        trsf.Invert()
        curve.Transform(trsf)

        # Always advance to next item
        edge_explorer.Next()
        
    return target_edge, adjacent_face

def create_profile_face(profile: Profile, sketch_plane: CoordSystem):
    """create a face from a sketch profile and the sketch plane"""
    origin = gp_Pnt(*(sketch_plane.origin*1.))
    normal = gp_Dir(*sketch_plane.normal)
    x_axis = gp_Dir(*sketch_plane.x_axis)
    gp_face = gp_Pln(gp_Ax3(origin, normal, x_axis))

    all_loops = [create_loop_3d(loop, sketch_plane) for loop in profile.children]
    topo_face = BRepBuilderAPI_MakeFace(gp_face, all_loops[0])
    for loop in all_loops[1:]:
        topo_face.Add(loop.Reversed())#
    return topo_face.Face()


def create_loop_3d(loop: Loop, sketch_plane: CoordSystem):
    """   create a 3D sketch loop   """
    topo_wire = BRepBuilderAPI_MakeWire()
    for curve in loop.children:
        topo_edge = create_edge_3d(curve, sketch_plane)
        if topo_edge == -1: # omitted
            continue
        topo_wire.Add(topo_edge)
    return topo_wire.Wire()


def create_edge_3d(curve: CurveBase, sketch_plane: CoordSystem):
    """create a 3D edge"""
    if isinstance(curve, Line):
        if np.allclose(curve.start_point, curve.end_point):
            return -1
        start_point = point_local2global(curve.start_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        topo_edge = BRepBuilderAPI_MakeEdge(start_point, end_point)
    elif isinstance(curve, Circle):
        center = point_local2global(curve.center, sketch_plane)
        axis = gp_Dir(*sketch_plane.normal)
        gp_circle = gp_Circ(gp_Ax2(center, axis), abs(float(curve.radius)))
        topo_edge = BRepBuilderAPI_MakeEdge(gp_circle)
    elif isinstance(curve, Arc):
        # print(curve.start_point, curve.mid_point, curve.end_point)
        start_point = point_local2global(curve.start_point, sketch_plane)
        mid_point = point_local2global(curve.mid_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        arc = GC_MakeArcOfCircle(start_point, mid_point, end_point).Value()
        topo_edge = BRepBuilderAPI_MakeEdge(arc)
    else:
        raise NotImplementedError(type(curve))
    return topo_edge.Edge()


def point_local2global(point, sketch_plane: CoordSystem, direction=False, to_gp_Pnt=True):
    """convert point in sketch plane local coordinates to global coordinates"""
    # mat_T = np.array([sketch_plane.x_axis,sketch_plane.y_axis,sketch_plane.normal])
    # r = R.from_euler('zyx', [sketch_plane._theta, sketch_plane._phi, sketch_plane._gamma], degrees=False)
    # mat_T = r.as_matrix().round(8)
    # point3d = np.array([point[0], point[1], 0.])
    # g_point = mat_T.dot((point3d - sketch_plane.trans))
    if direction:
        g_point = point[0] * sketch_plane.x_axis + point[1] * sketch_plane.y_axis
    else:
        g_point = point[0] * sketch_plane.x_axis + point[1] * sketch_plane.y_axis + sketch_plane.origin
    if to_gp_Pnt:
        return gp_Pnt(*g_point)
    return g_point.round(7)

def CADsolid2pc(shape, n_points, name=None):
    """convert opencascade solid to point clouds"""
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    if bbox.IsVoid():
        raise ValueError("box check failed")

    if name is None:
        name = random.randint(100000, 999999)
    write_stl_file(shape, "tmp_out_{}.stl".format(name))
    out_mesh = trimesh.load("tmp_out_{}.stl".format(name))
    os.system("rm tmp_out_{}.stl".format(name))
    out_pc, _ = sample_surface(out_mesh, n_points)
    return out_pc

def export_step_file(shape, filedir, filename):
    if shape is None:
        raise ValueError("The provided shape is None.")

    # initialize the STEP exporter
    step_writer = STEPControl_Writer()
    dd = step_writer.WS().TransferWriter().FinderProcess()
    Interface_Static_SetCVal("write.step.schema", "AP203")

    # transfer shapes and write file
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    filepath = str(Path(filedir) / filename)
    step_writer.Transfer(shape, STEPControl_AsIs)
    status = step_writer.Write(filepath)

    if status != IFSelect_RetDone:
        raise AssertionError("load failed")
    print("save step file to", filepath)

def apply_fillet_with_timeout(fillet, radius1, radius2, is_symmetric, edge, timeout=5, use_timeout=False):
    """Apply fillet with timeout"""
    def apply_fillet(fillet, radius1, radius2, is_symmetric, edge, result_queue):
        try:
            assert edge, "No edge defined."
            fillet.Add(radius1, radius2, edge)
            fillet.Build()
            assert fillet.IsDone(), "Fillet operation did not complete successfully."
            shape = fillet.Shape()
            result_queue.put((True, shape))
        except Exception as e:
            print(f"Error in apply_fillet: {e}")
            result_queue.put((False, None))
    
    # If not using timeout, apply fillet directly
    if not use_timeout:
        try:
            assert edge, "No edge defined."
            fillet.Add(radius1, radius2, edge)
            fillet.Build()
            assert fillet.IsDone(), "Fillet operation did not complete successfully."
            shape = fillet.Shape()
            return True, shape
        except Exception as e:
            print(f"Error in apply_fillet: {e}")
            return False, None
    
    # Create a queue for the result
    result_queue = multiprocessing.Queue()
    
    # Create and start the process
    process = multiprocessing.Process(
        target=apply_fillet, 
        args=(fillet, radius1, radius2, is_symmetric, edge, result_queue)
    )
    process.start()
    
    # Wait for the result with timeout
    process.join(timeout)
    
    # If process is still alive after timeout
    if process.is_alive():
        print(f"Fillet.Add timed out after {timeout} seconds")
        process.terminate()
        process.join()
        return False, None
    
    # Get the result if available
    if not result_queue.empty():
        return result_queue.get()
    return False, None

def apply_chamfer_with_timeout(chamfer, distance1, distance2, edge, timeout=5, face=None, use_timeout=False):
    """Apply chamfer with timeout, with optional angle parameter"""
    def apply_chamfer(chamfer, distance1, distance2, edge, result_queue, face=None):
        try:
            # If angle is provided, use it to create an angled chamfer
            if face is not None:
                chamfer.Add(distance1, distance2, edge, face)
            else:
                # Simple distance chamfer
                chamfer.Add(distance1, edge)
                
            chamfer.Build()
            assert chamfer.IsDone(), "Chamfer operation did not complete successfully."
            shape = chamfer.Shape()
            result_queue.put((True, shape))
        except Exception as e:
            print(f"Error in apply_chamfer: {e}")
            result_queue.put((False, None))
    
    # If not using timeout, apply chamfer directly
    if not use_timeout:
        try:
            # If angle is provided, use it to create an angled chamfer
            if face is not None:
                chamfer.Add(distance1, distance2, edge, face)
            else:
                # Simple distance chamfer
                chamfer.Add(distance1, edge)
                
            chamfer.Build()
            assert chamfer.IsDone(), "Chamfer operation did not complete successfully."
            shape = chamfer.Shape()
            return True, shape
        except Exception as e:
            print(f"Error in apply_chamfer: {e}")
            return False, None
    
    # Create a queue for the result
    result_queue = multiprocessing.Queue()
    
    # Create and start the process
    process = multiprocessing.Process(
        target=apply_chamfer, 
        args=(chamfer, distance1, distance2, edge, result_queue, face)
    )
    process.start()
    
    # Wait for the result with timeout
    process.join(timeout)
    
    # If process is still alive after timeout
    if process.is_alive():
        print(f"Chamfer.Add timed out after {timeout} seconds")
        process.terminate()
        process.join()
        return False, None
    
    # Get the result if available
    if not result_queue.empty():
        return result_queue.get()
    return False, None
