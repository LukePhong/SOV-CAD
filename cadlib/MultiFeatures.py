import numpy as np
import random
from .sketch import Profile
from .macro import *
from .math_utils import cartesian2polar, polar2cartesian, polar_parameterization, polar_parameterization_inverse
from scipy.spatial.transform import Rotation as R
from copy import deepcopy


class CoordSystem(object):
    """Local coordinate system for sketch plane."""
    def __init__(self, origin, theta, phi, gamma, y_axis=None, is_numerical=False):
        self.origin = origin
        ## Default external rotation order is 'z-y-x'
        self._theta = theta # -pi~pi
        self._phi = phi     # -pi/2~pi
        self._gamma = gamma # -pi~pi
        self._y_axis = y_axis
        self.is_numerical = is_numerical
        # Tolerance for zero values
        self._zero_tolerance = 0.1

    # @property
    # def origin(self):
    #     r = R.from_euler('zyx', [self._theta, self._phi, self._gamma], degrees=False)
    #     r_mat = r.as_matrix().round(8)
    #     l_origin = np.array([0., 0., 0.])
    #     w_origin = r_mat.dot(l_origin) + self.trans
    #     return w_origin

    @property
    def normal(self):
        # return polar2cartesian([self._theta, self._phi])
        r = R.from_euler('zyx',[self._theta, self._phi, self._gamma],degrees=False)
        r_mat = r.as_matrix().round(8)
        n_origin = np.array([0.,0.,1.])
        n_normal = r_mat.dot(n_origin)
        # Clean near-zero values to prevent floating point errors
        n_normal = np.where(np.abs(n_normal) < self._zero_tolerance, 0.0, n_normal)
        return n_normal
        
    @property
    def x_axis(self):
        # normal_3d, x_axis_3d = polar_parameterization_inverse(self._theta, self._phi, self._gamma)
        # return x_axis_3d
        r = R.from_euler('zyx', [self._theta, self._phi, self._gamma], degrees=False)
        r_mat = r.as_matrix().round(8)
        x_origin = np.array([1., 0., 0.])
        x_axis_3d = r_mat.dot(x_origin)
        # Clean near-zero values to prevent floating point errors
        x_axis_3d = np.where(np.abs(x_axis_3d) < self._zero_tolerance, 0.0, x_axis_3d)
        return x_axis_3d


    @property
    def y_axis(self):
        # if self._y_axis is None:
        y_axis_3d = np.cross(self.normal, self.x_axis)
        # Clean near-zero values to prevent floating point errors
        y_axis_3d = np.where(np.abs(y_axis_3d) < self._zero_tolerance, 0.0, y_axis_3d)
        return y_axis_3d
        # return polar2cartesian(self._y_axis)

    @staticmethod
    def from_dict(stat):
        # origin represents translation, general order is scale-rotation-translation
        if 'Rotation' in stat:
            origin = (np.array(stat['origin'])*1000).round(8)
            matrix = np.array(stat['Rotation']).round(8).T
        else:
            # Handle format with x_axis, y_axis, z_axis dictionaries
            origin = np.array([stat['origin']['x'], stat['origin']['y'], stat['origin']['z']]) * 1000
            origin = origin.round(8)
            
            # Extract axis vectors and build rotation matrix
            x_axis = np.array([stat['x_axis']['x'], stat['x_axis']['y'], stat['x_axis']['z']])
            y_axis = np.array([stat['y_axis']['x'], stat['y_axis']['y'], stat['y_axis']['z']])
            z_axis = np.array([stat['z_axis']['x'], stat['z_axis']['y'], stat['z_axis']['z']])
            matrix = np.column_stack([x_axis, y_axis, z_axis]).round(8)
        
        # Common transformation logic
        model2sketch = np.zeros((4, 4))
        model2sketch[:3, :3] = matrix
        model2sketch[:3, 3] = origin
        model2sketch[3, 3] = 1
        sketch2model = np.linalg.inv(model2sketch)
        matrix = sketch2model[:3, :3]
        origin = sketch2model[:3, 3]
        r = R.from_matrix(matrix)
        eular = r.as_euler('zyx', degrees=False)
        theta, phi, gamma = eular
        return CoordSystem(origin, theta, phi, gamma)
        # return CoordSystem(origin, 0, 0, 0)

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        origin = vec[:3]
        theta, phi, gamma = vec[3:]
        system = CoordSystem(origin, theta, phi, gamma)
        # print(system)
        if is_numerical:
            system.denumericalize(n)
        return system

    def __str__(self):
        return "origin: {}, normal: {}, x_axis: {}, y_axis: {}, theta:{}, phi:{}, gamma:{}".format(
            self.origin.round(4), self.normal.round(4), self.x_axis.round(4), self.y_axis.round(4),
            round(self._theta,4), round(self._phi, 4), round(self._gamma))

    def transform(self, translation, scale):
        self.origin = (self.origin + translation) * scale

    def numericalize(self, n=256):
        """NOTE: shall only be called after normalization"""
        # assert np.max(self.origin) <= 1.0 and np.min(self.origin) >= -1.0 # TODO: origin can be out-of-bound!
        ## theta: -180~180 phi: -90~180 gamma: -180~180
        self.origin = ((self.origin + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self._theta = ((self._theta / np.pi + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self._phi = ((self._phi*2 / (3 * np.pi) + 1/3)*n).round().clip(min=0, max=n).astype(np.int)
        self._gamma = ((self._gamma / np.pi + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self.is_numerical = True

    def denumericalize(self, n=256):
        self.origin = self.origin / n * 2 - 1.0
        # tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta = (self._theta / n * 2 - 1.0) * np.pi
        self._phi = (self._phi / n - 1/3) * (3*np.pi) / 2.0
        self._gamma = (self._gamma / n * 2 - 1.0) * np.pi
        self.is_numerical = False

    def to_vector(self):
        return np.array([*self.origin, self._theta, self._phi, self._gamma])

    def flip_normal(self):
        """Flip the direction of the normal vector by modifying the internal Euler angles."""
        # Add π to _phi to flip the normal direction
        # Since _phi is constrained to -π/2 to π, we need to handle wrap-around
        self._phi = self._phi + np.pi
        
        # Ensure _phi stays within valid range [-π/2, π]
        # If _phi exceeds π, we wrap it back and adjust _theta
        if self._phi > np.pi:
            self._phi = self._phi - 2*np.pi
            # Also flip _theta by π to maintain the same final orientation
            self._theta = self._theta + np.pi
            # Keep _theta in range [-π, π]
            if self._theta > np.pi:
                self._theta = self._theta - 2*np.pi
            elif self._theta < -np.pi:
                self._theta = self._theta + 2*np.pi


class Extrude(object):
    """Single extrude operation with corresponding a sketch profile.
    NOTE: only support single sketch profile. Extrusion with multiple profiles is decomposed."""
    def __init__(self, profile: Profile, sketch_plane: CoordSystem,
                 extent_one, extent_two, sketch_pos, sketch_size, type):
        """
        Args:
            profile (Profile): normalized sketch profile
            sketch_plane (CoordSystem): coordinate system for sketch plane
            operation (int): index of EXTRUDE_OPERATIONS, see macro.py
            extent_type (int): index of EXTENT_TYPE, see macro.py
            extent_one (float): extrude distance in normal direction (NOTE: it's negative in some data)
            extent_two (float): extrude distance in opposite direction
            sketch_pos (np.array): the global 3D position of sketch starting point
            sketch_size (float): size of the sketch
        """
        self.profile = profile # normalized sketch
        self.sketch_plane = sketch_plane
        # self.extent_type = extent_type
        self.extent_one = extent_one
        self.extent_two = extent_two

        # self.isreversed = isreversed
        self.sketch_pos = sketch_pos
        self.sketch_size = sketch_size
        self.type = type ## Used to determine whether it's Cut or Boss, 1:Cut, 0:Boss

    @staticmethod
    def from_dict(all_stat, extrude_id, sketch_dim=256):
        """construct Extrude from json data

        Args:
            all_stat (dict): all json data
            extrude_id (str): entity ID for this extrude
            sketch_dim (int, optional): sketch normalization size. Defaults to 256.

        Returns:
            list: one or more Extrude instances
        """
        extrude_entity = all_stat["entities"][extrude_id]
        if "Invalid" in extrude_entity.keys():
            return None

        all_skets = []
        # Handle both "Profiles" and "profiles" keys
        profiles_key = "Profiles" if "Profiles" in extrude_entity else "profiles"
        n = len(extrude_entity[profiles_key])

        if "extent_two" not in extrude_entity.keys():
            # operation = EXTRUDE_OPERATIONS.index(extrude_entity["operation"])
            type = float(extrude_entity["type"] == "ExtrusionCut")
            extent_two = 0.0
            # if extrude_entity["extent_one"]["distance"]["forward condition"] == "ThroughAll":
            #     box = all_stat["properties"]["bounding_box"]
            #     max_point = box["max_point"]
            #     min_point = box["min_point"]
            #     extent_one = 1000 * 2 * np.max(np.abs([max_point["x"],min_point["x"],max_point["y"],min_point["y"],max_point["z"],min_point["z"]]))
            #     # extent_one = 1000 * np.linalg.norm([max_point["x"]-min_point["x"],max_point["y"]-min_point["y"],max_point["z"]-min_point["z"]])
            # else:
            extent_one = 1000 * extrude_entity["extent_one"]["distance"]["forward distance"]

            isreversed = int(extrude_entity["extent_one"]["distance"]["IsReversed"])
            ## extent_two is only calculated when it's BothSide
            # if extrude_entity["extent_type"] == "BothSidesFeatureExtentType":
            # if extrude_entity["extent_one"]["distance"]["reverse condition"] == "ThroughAll":
            #     box = all_stat["properties"]["bounding_box"]
            #     max_point = box["max_point"]
            #     min_point = box["min_point"]
            #     extent_two = 1000 * 2 * np.max(np.abs([max_point["x"],min_point["x"],max_point["y"],min_point["y"],max_point["z"],min_point["z"]]))
            # else:
            extent_two = 1000 * extrude_entity["extent_one"]["distance"]["reverse distance"]

            # extent_type = EXTENT_TYPE.index(extrude_entity["extent_type"])
            forward_condition = extrude_entity["extent_one"]["distance"]["forward condition"]
            reverse_condition = extrude_entity["extent_one"]["distance"]["reverse condition"]
            unchanged_condition = ["Up to Vertex", "Up to Surface"]
            if(isreversed == 1 and (forward_condition not in unchanged_condition) and (reverse_condition not in unchanged_condition)):
                extent_one, extent_two = extent_two, extent_one
        else:
            # operation = EXTRUDE_OPERATIONS.index(extrude_entity["operation"])
            type = float(extrude_entity["operation"] == "CutFeatureOperation")
            # extent_type = EXTENT_TYPE.index(extrude_entity["extent_type"])
            extent_one = 1000 * extrude_entity["extent_one"]["distance"]["value"]
            extent_two = 0.0
            if extrude_entity["extent_type"] == "TwoSidesFeatureExtentType":
                extent_two = 1000 * extrude_entity["extent_two"]["distance"]["value"]
            if extrude_entity["extent_type"] == "SymmetricFeatureExtentType":
                extent_two = extent_one

            # if operation == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation"):
            #     all_operations = [operation] + [EXTRUDE_OPERATIONS.index("JoinFeatureOperation")] * (n - 1)
            # else:
            #     all_operations = [operation] * n


        for i in range(len(extrude_entity[profiles_key])):
            sket_id, profile_id = extrude_entity[profiles_key][i]["sketch"], extrude_entity[profiles_key][i]["profile"]
            sket_entity = all_stat["entities"][sket_id]
            if  profile_id in sket_entity["profiles"].keys():
                sket_profile = Profile.from_dict(sket_entity["profiles"][profile_id])
            elif profile_id in sket_entity["contours"].keys():
                sket_profile = Profile.from_dict(sket_entity["contours"][profile_id])
            else:
                sket_profile = Profile.from_dict(sket_entity["defaultProfiles"][profile_id])
            if sket_profile is None:
                return None
            sket_plane = CoordSystem.from_dict(sket_entity["transform"])

            if "extent_two" in extrude_entity.keys() and type == 1:
                sket_plane.flip_normal()

            # normalize profile
            point = sket_profile.start_point

            sket_pos = point[0] * sket_plane.x_axis + point[1] * sket_plane.y_axis + sket_plane.origin
            # sket_pos = sket_plane.origin
            sket_size = sket_profile.bbox_size
            sket_profile.normalize(size=256)
            all_skets.append((sket_profile, sket_plane, sket_pos, sket_size))

        # extent_n may be negative
        extent_one = abs(extent_one)
        extent_two = abs(extent_two)
        
        ## Returns a list corresponding to an Extrusion, where each instance in the list corresponds to each Profile in the Sketch
        ## That is, one Profile corresponds to one Extrusion
        return [Extrude(all_skets[i][0], all_skets[i][1], extent_one, extent_two,
                        all_skets[i][2], all_skets[i][3], type) for i in range(n)]

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        assert (vec[-1][0] == EXT_IDX or vec[-1][0] == EXTCUT_IDX) and vec[0][0] == SOL_IDX, "Extrude vector is not valid"
        if vec[-1][0] == EXT_IDX:
            type = 0
        else:
            type = 1
        profile_vec = np.concatenate([vec[:-1], EOS_VEC[np.newaxis]])
        profile = Profile.from_vector(profile_vec, is_numerical=is_numerical)
        # ext_vec = vec[-1][-N_ARGS_EXT:]
        ### Considering the influence of sketch plane's origin
        ext_vec = vec[-1][N_ARGS_SKETCH+1:(N_ARGS_SKETCH+N_ARGS_PLANE+N_ARGS_EXT_PARAM+1)]

        # sket_pos = ext_vec[N_ARGS_PLANE:N_ARGS_PLANE + 3]
        sket_pos = ext_vec[N_ARGS_ROTAT:N_ARGS_ROTAT + 3]
        sket_size = ext_vec[N_ARGS_PLANE - 1]
        # print(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        sket_plane = CoordSystem.from_vector(np.concatenate([sket_pos, ext_vec[:N_ARGS_ROTAT]]))

        ext_param = ext_vec[-N_ARGS_EXT_PARAM:]

        res = Extrude(profile, sket_plane, ext_param[0], ext_param[1],
                      sket_pos, sket_size, type)
        if is_numerical:
            res.denumericalize(n)
        return res

    def __str__(self):
        s = "Sketch-Extrude pair:"
        s += "\n  -" + str(self.sketch_plane)
        s += "\n  -sketch position: {}, sketch size: {}".format(self.sketch_pos.round(4), round(self.sketch_size,4))
        s += "\n  -type:{}, extent_one:{}, extent_two:{}".format(
            self.type, round(self.extent_one,4), round(self.extent_two,4))
        s += "\n  -" + str(self.profile)
        return s

    def transform(self, translation, scale):
        """linear transformation"""
        # self.profile.transform(np.array([0, 0]), scale)
        self.sketch_plane.transform(translation, scale)
        self.extent_one *= scale
        self.extent_two *= scale
        self.sketch_pos = (self.sketch_pos + translation) * scale
        self.sketch_size *= scale

    def numericalize(self, n=256):
        """quantize the representation.
        NOTE: shall only be called after CADSequence.normalize (the shape lies in unit cube, -1~1)"""
        if(self.type == 1 and self.extent_one>=2):
            self.extent_one = 2
        if(self.type == 1 and self.extent_two>=2):
            self.extent_two = 2
        assert 0 <= self.extent_one <= 2.0 and 0 <= self.extent_two <= 2.0
        self.profile.numericalize(n)
        self.sketch_plane.numericalize(n)
        if(self.type == 1):
            self.extent_one = np.clip(np.ceil(self.extent_one / 2 * n), a_min=0, a_max=n).astype(np.int)
            self.extent_two = np.clip(np.ceil(self.extent_two / 2 * n), a_min=0, a_max=n).astype(np.int)
        else:
            self.extent_one = np.clip(round(self.extent_one / 2 * n), a_min=0, a_max=n).astype(np.int)
            self.extent_two = np.clip(round(self.extent_two / 2 * n), a_min=0, a_max=n).astype(np.int)
        # self.isreversed = int(self.isreversed)
        # self.extent_type = int(self.extent_type)

        self.sketch_pos = ((self.sketch_pos + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self.sketch_size = np.clip(round(self.sketch_size / 2 * n), a_min=0, a_max=n).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.extent_one = self.extent_one / n * 2
        self.extent_two = self.extent_two / n * 2
        self.sketch_plane.denumericalize(n)
        self.sketch_pos = self.sketch_pos / n * 2 - 1.0
        self.sketch_size = self.sketch_size / n * 2

        # self.isreversed = self.isreversed
        # self.extent_type = self.extent_type

    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        profile_vec = self.profile.to_vector(max_n_loops, max_len_loop, pad=False)
        if profile_vec is None:
            return None
        
        sket_plane_orientation = self.sketch_plane.to_vector()[3:]
        ext_param = list(sket_plane_orientation) + list(self.sketch_pos) + [self.sketch_size] + \
                    [self.extent_one, self.extent_two]##, self.isreversed , self.extent_type
        if self.type == 0:
            ext_vec = np.array([EXT_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *ext_param, *[PAD_VAL]*(N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)])
        else:
            ext_vec = np.array([EXTCUT_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *ext_param, *[PAD_VAL] * (N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)])
        vec = np.concatenate([profile_vec[:-1], ext_vec[np.newaxis], profile_vec[-1:]], axis=0) # NOTE: last one is EOS
        if pad:
            pad_len = max_n_loops * max_len_loop - vec.shape[0]
            vec = np.concatenate([vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return vec

class Axis():

    def __init__(self, cent=None, direction=None, sketch_plane:CoordSystem=None):
        self.cent = cent
        self.direction = direction
        if(self.direction is not None):
            self.direction_angle = self.get_direction_angle()## 0~2*pi
        self.sketch_plane = sketch_plane
        if(self.sketch_plane is not None):
            self.cent = np.array([0, 0])
            self.direction = np.array([1, 0])
            self.direction_angle = 0

    def get_direction_angle(self):
        ## direction[0] = cos(theta)， direction[1]=sin(theta)
        angle = np.arccos(self.direction[0])
        if(self.direction[1]<0):
            angle = np.pi*2 - angle
        return angle

    @staticmethod
    def get_direction(p1, p2):
        vec1_2 = p2-p1
        norm = np.linalg.norm(vec1_2,ord=2)
        return vec1_2 / norm

    @staticmethod
    def from_dict(stat):
        ## Here the default z coordinate is 0, equivalent to being on the same plane as the sketch
        start_point = 1000 * np.array([stat["start_point"]["x"], stat["start_point"]["y"]])
        end_point = 1000 * np.array([stat["end_point"]["x"], stat["end_point"]["y"]])
        cent = (start_point + end_point) / 2
        direction = Axis.get_direction(start_point, end_point)
        return Axis(cent, direction)

    def transform(self, translation, scale):
        self.sketch_plane.transform(translation, scale)

    def normalize(self, sketch_plane:CoordSystem):
        # print(self.cent)
        # print(self.direction)
        # print(self.direction_angle)
        # input()
        cur_sketch_plane = CoordSystem(sketch_plane.origin, sketch_plane._theta, sketch_plane._phi, sketch_plane._gamma)
        ## Determine whether the cent of Axis is the origin
        if(self.cent[0] ==0 and self.cent[1] == 0):
            trans = np.array([0,0,0])
        ## Determine whether Axis is perpendicular to x-axis
        elif(self.direction[0] == 0):
            x = self.cent[0]
            y = 0
            trans = np.array([x,y,0])
        elif(self.direction[1] == 0):
            x = 0
            y = self.cent[1]
            trans = np.array([x,y,0])
        ## Determine whether Axis passes through the origin
        elif((self.cent[1] / self.cent[0]) == (self.direction[1] / self.direction[0])):
            trans = np.array([0,0,0])
        ## First find the projection coordinates of the origin on the line. Axis is a line passing through cent with slope K. We need to find a line passing through the origin and perpendicular to Axis, where k1*k2=-1
        else:
            k1 = self.direction[1] / self.direction[0]
            k2 = -1 / k1
            ## The line passing through the origin and perpendicular to Axis is y = K2x. By solving the two equations simultaneously, we can find the intersection point
            x = (k1*self.cent[0] - self.cent[1]) / (k1 -k2)
            y = k2 * x
            trans = np.array([x,y,0])

        ## This is equivalent to a two-layer coordinate system transformation. First, rotate and translate the Axis coordinates to the sketch coordinate system
        r1 = R.from_euler("zyx", [self.direction_angle, 0, 0], degrees=False)
        rotate_matrix = r1.as_matrix()
        trans_matrix = trans
        axis2sketch = np.zeros((4, 4))
        axis2sketch[:3, :3] = rotate_matrix
        axis2sketch[:3, 3] = trans_matrix
        axis2sketch[3, 3] = 1
        ## Then transform the sketch coordinate system to the World coordinate system
        r2 = R.from_euler('zyx', [cur_sketch_plane._theta, cur_sketch_plane._phi, cur_sketch_plane._gamma], degrees=False)
        r_matrix = r2.as_matrix()
        t_matrix = cur_sketch_plane.origin
        sketch2model = np.zeros((4, 4))
        sketch2model[:3, :3] = r_matrix
        sketch2model[:3, 3] = t_matrix
        sketch2model[3, 3] = 1
        ## Combine the matrices
        Rotation_T = np.matmul(sketch2model, axis2sketch)
        ## Decompose again
        matrix = Rotation_T[:3, :3]
        origin = Rotation_T[:3, 3]
        r = R.from_matrix(matrix)
        eular = r.as_euler('zyx', degrees=False)
        theta, phi, gamma = eular
        self.sketch_plane = CoordSystem(origin, theta, phi, gamma)

        ## After rotation and translation, Axis is at the origin position with direction [1,0]
        self.cent = np.array([0,0])
        self.direction = np.array([1,0])
        self.direction_angle = 0

    def to_vector(self):
        sketch_pos = list(self.sketch_plane.to_vector()[:3])
        sket_plane_orientation = list(self.sketch_plane.to_vector()[3:])
        vec = np.array(
            [REVA_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *sket_plane_orientation, *sketch_pos, *[PAD_VAL] * (1 + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)])
        return vec

    @staticmethod
    def from_vector(vec):
        plane_vec = vec[N_ARGS_SKETCH:(N_ARGS_SKETCH+N_ARGS_PLANE)]
        plane_pos = plane_vec[N_ARGS_ROTAT:N_ARGS_ROTAT+3]
        plane_orientation = plane_vec[:N_ARGS_ROTAT]
        # print(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        sket_plane = CoordSystem.from_vector(np.concatenate([plane_pos, plane_orientation]))
        return Axis(sketch_plane=sket_plane)

    def __str__(self):
        return self.sketch_plane.__str__()


    def numericalize(self, n=256):
        # self.cent = ((self.cent + 1.0) / 2 * n).round().clip(min=0, max = n-1).astype(np.int)
        self.sketch_plane.numericalize(n)

    def denumericalize(self, n=256):
        # self.cent = self.cent / n * 2 - 1.0
        self.sketch_plane.denumericalize(n)



class Revolve(object):
    """Single extrude operation with corresponding a sketch profile.
    NOTE: only support single sketch profile. Extrusion with multiple profiles is decomposed."""
    def __init__(self, profile: Profile, sketch_plane: CoordSystem,
                 Axis, Angle, sketch_pos, sketch_size, type):
        """
        Args:
            profile (Profile): normalized sketch profile
            sketch_plane (CoordSystem): coordinate system for sketch plane
            operation (int): index of Revolve_OPERATIONS, see macro.py
            Axis:
            Angle:
            sketch_pos (np.array): the global 3D position of sketch starting point
            sketch_size (float): size of the sketch
        """
        self.profile = profile # normalized sketch
        self.sketch_plane = sketch_plane # transform
        self.axis = Axis
        self.angle = Angle
        self.sketch_pos = sketch_pos # origin
        self.sketch_size = sketch_size
        self.type = type

    @staticmethod
    def get_operation_type(revovle_entity):
        if (revovle_entity["type"] == "Revolution"):
            operation_type = "NewBodyFeatureOperation"
        elif(revovle_entity["type"] == "RevCut"):
            operation_type = "CutFeatureOperation"
        return operation_type

    @staticmethod
    def from_dict(all_stat, revovle_id, sketch_dim=256):
        """construct Extrude from json data
        Args:
            all_stat (dict): all json data
            revolve_id (str): entity ID for this extrude
            sketch_dim (int, optional): sketch normalization size. Defaults to 256.
        Returns:
            list: one or more Revolve instances
        """
        revolve_entity = all_stat["entities"][revovle_id]
        all_skets = []
        n = len(revolve_entity["Profiles"])
        ##assert n == 1, "Revolve profile num need to be 1"
        axis = Axis.from_dict(revolve_entity['Axis'])
        ### After testing, generally Revolve has only one Profile, which is the difference from Extrude
        for i in range(len(revolve_entity["Profiles"])):
            sket_id, profile_id = revolve_entity["Profiles"][i]["sketch"], revolve_entity["Profiles"][i]["profile"]
            sket_entity = all_stat["entities"][sket_id]
            if profile_id in sket_entity["profiles"].keys():
                sket_profile = Profile.from_dict(sket_entity["profiles"][profile_id])
            else:
                sket_profile = Profile.from_dict(sket_entity["contours"][profile_id])
            if sket_profile is None:
                return None
            sket_plane = CoordSystem.from_dict(sket_entity["transform"])
            # normalize profile
            point = sket_profile.start_point

            sket_pos = point[0] * sket_plane.x_axis + point[1] * sket_plane.y_axis + sket_plane.origin
            # sket_pos = sket_plane.origin
            sket_size = sket_profile.bbox_size
            sket_profile.normalize(size=256)
            all_skets.append((sket_profile, sket_plane, sket_pos, sket_size))
        # normalize axis
        axis.normalize(sket_plane)
        type = float(revolve_entity["type"] == "RevCut")

        ## Returns a list corresponding to an Extrusion, where each instance in the list corresponds to each Profile in the Sketch
        ## That is, one Profile corresponds to one Extrusion
        Angle = revolve_entity['Angle']
        return [Revolve(all_skets[i][0], all_skets[i][1], axis, Angle,
                        all_skets[i][2], all_skets[i][3], type) for i in range(n)]

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        """vector representation: commands [SOL, ..., SOL, ..., Axis, Revolve]
        revolve vector: [Profile_vec, Axis_vec, Revolve_vec]
        """
        assert (vec[-1][0] == REV_IDX or vec[-1][0] == REVCUT_IDX) and vec[0][0] == SOL_IDX and vec[-2][0] == REVA_IDX, "Revolve vector format error"
        profile_vec = np.concatenate([vec[:-2], EOS_VEC[np.newaxis]])
        profile = Profile.from_vector(profile_vec, is_numerical=is_numerical)

        if vec[-1][0] == REV_IDX:
            type = 0
        else:
            type = 1

        rev_vec = vec[-1][1:]
        # sket_pos = ext_vec[N_ARGS_PLANE:N_ARGS_PLANE + 3]
        sket_pos = rev_vec[(N_ARGS_SKETCH+N_ARGS_ROTAT):(N_ARGS_SKETCH+N_ARGS_ROTAT+3)]
        sket_size = rev_vec[N_ARGS_SKETCH + N_ARGS_PLANE - 1]
        sketch_plane_orientation = rev_vec[N_ARGS_SKETCH : (N_ARGS_SKETCH + N_ARGS_ROTAT)]
        # print(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        # sket_plane = CoordSystem.from_vector(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        ## Because the plane's origin was added during encapsulation, there's no need to use sketch pos to replace origin here
        sket_plane = CoordSystem.from_vector(np.concatenate([sket_pos, sketch_plane_orientation]))
        axis_vec = vec[-2][1:]
        axis = Axis.from_vector(axis_vec)
        angle = rev_vec[N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM - 1]
        res = Revolve(profile, sket_plane, axis, angle, sket_pos, sket_size, type)
        if is_numerical:
            res.denumericalize(n)
        return res

    def __str__(self):
        s = "Sketch-Revolve pair:"
        s += "\n  -" + str(self.sketch_plane)
        s += "\n  -sketch position: {}, sketch size: {}".format(self.sketch_pos.round(8), round(self.sketch_size,8))
        s += "\n  -Axis:{}, ".format(self.axis)
        s += "\n  -" + str(self.profile)
        return s

    def transform(self, translation, scale):
        """linear transformation"""
        # self.profile.transform(np.array([0, 0]), scale)
        self.sketch_plane.transform(translation, scale)
        self.axis.transform(translation, scale)
        self.sketch_pos = (self.sketch_pos + translation) * scale
        self.sketch_size *= scale

    def numericalize(self, n=256):
        """quantize the representation.
        NOTE: shall only be called after CADSequence.normalize (the shape lies in unit cube, -1~1)"""
        self.profile.numericalize(n)
        self.sketch_plane.numericalize(n)
        self.axis.numericalize(n)
        self.angle = np.clip(round(self.angle / (np.pi*2) * n), a_min=0, a_max=n).astype(np.int)

        self.sketch_pos = ((self.sketch_pos + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self.sketch_size = np.clip(round(self.sketch_size / 2 * n), a_min=0, a_max=n).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.sketch_plane.denumericalize(n)
        self.axis.denumericalize(n)
        self.sketch_pos = self.sketch_pos / n * 2 - 1.0
        self.sketch_size = self.sketch_size / n * 2
        self.angle = self.angle / n * 2 *np.pi


    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        profile_vec = self.profile.to_vector(max_n_loops, max_len_loop, pad=False)
        if profile_vec is None:
            return None
        
        sket_plane_orientation = self.sketch_plane.to_vector()[3:]
        axis_vector = self.axis.to_vector()
        sketch_param = list(sket_plane_orientation) + list(self.sketch_pos) + [self.sketch_size]
        rev_param = [self.angle]
        if self.type == 0:
            rev_vec = np.array([REV_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *sketch_param, *[PAD_VAL]*N_ARGS_EXT_PARAM,
                            *rev_param, *[PAD_VAL]*N_ARGS_FILL_PARAM])
        else:
            rev_vec = np.array([REVCUT_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *sketch_param, *[PAD_VAL]*N_ARGS_EXT_PARAM,
                            *rev_param, *[PAD_VAL]*N_ARGS_FILL_PARAM])
        vec = np.concatenate([profile_vec[:-1], axis_vector[np.newaxis], rev_vec[np.newaxis], profile_vec[-1:]], axis=0) # NOTE: last one is EOS
        if pad:
            pad_len = max_n_loops * max_len_loop - vec.shape[0]
            vec = np.concatenate([vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return vec

class Fillet():
    def __init__(self, cent:list, d1, d2, is_sym=True):
        self.cent = cent
        # for cent in cents:
        #     self.cents.append([i * 1000 for i in cent])
        self.d1 = d1
        self.d2 = d2
        self.is_sym = is_sym

    @staticmethod
    def from_dict(all_stat, fillet_id):
        fillet_entity = all_stat["entities"][fillet_id]
        cents = fillet_entity['Edges']
        d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
        d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
        is_sym = fillet_entity['Distance Paramters']['isSymmetric']
        return [Fillet([1000* i for i in cent], d1, d2, is_sym) for cent in cents]

    def __str__(self):
        s =  "Fillet Edges: \n"
        s += "x: {}, y: {}, z:{}\n".format(round(self.cent[0], 8), round(self.cent[1], 8), round(self.cent[2], 8))
        s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
        return s

    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        fillet_vec = np.array(
            [FILLET_IDX, *[PAD_VAL] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1, self.d2])
        vec = np.stack([fillet_vec, EOS_VEC], axis=0)
        return vec

    def transform(self, translation, scale):
        self.d1 *= scale
        self.d2 *= scale
        self.cent = list((np.array(self.cent) + translation) * scale)

    def numericalize(self, n=256):
        """quantize the representation."""
        self.cent = list(((np.array(self.cent) + 1.0) / 2 *n).round().clip(min=0, max=n).astype(np.int))
        self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self.d2 = (self.d2 / 2 * n).round().clip(min=0, max=n).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.cent = list(np.array(self.cent) / n * 2 - 1.0)
        self.d1 = self.d1 / n * 2
        self.d2 = self.d2 / n * 2

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        fillet_vec = vec[-1]
        assert fillet_vec[0] == FILLET_IDX
        cent = fillet_vec[-N_ARGS_FILL_PARAM:-2]
        d1 = fillet_vec[-2]
        d2 = fillet_vec[-1]
        res =  Fillet(cent, d1, d2)
        if is_numerical:
            res.denumericalize(n)
        return res

class Chamfer():
    def __init__(self, cent:list, d1, d2=None, angle=None):
        self.cent = cent
        # for cent in cents:
        #     self.cents.append([i * 1000 for i in cent])
        self.d1 = d1
        self.angle = angle
        if d2 is None:
            self.d2 = round(self.d1 * np.tan(self.angle),8)
        else:
            self.d2 = d2

    @staticmethod
    def from_dict(all_stat, chamfer_id):
        ## cents is a list
        fillet_entity = all_stat["entities"][chamfer_id]
        cents = fillet_entity['Edges']
        AD_type = fillet_entity['Distance Paramters']['type']
        if AD_type == 'ChamferAngleDistance':
            d1 = 1000 * fillet_entity['Distance Paramters']['Distance']
            d2 = None
            angle = fillet_entity['Distance Paramters']['Angle']
        elif AD_type == 'ChamferDistanceDistance':
            d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
            d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
            angle = None
        else:
            raise Exception('Invalid Chamfer Distance Parameters')
        return [Chamfer([1000* i for i in cent], d1, d2, angle) for cent in cents]

    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        chamfer_vec = np.array(
            [CHAMFER_IDX, *[PAD_VAL] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1, self.d2])
        vec = np.stack([chamfer_vec, EOS_VEC], axis=0)
        return vec

    def transform(self, translation, scale):
        self.d1 *= scale
        self.d2 *= scale
        self.cent = list((np.array(self.cent) + translation) * scale)

    def numericalize(self, n=256):
        """quantize the representation."""
        self.cent = list(((np.array(self.cent) + 1.0) / 2 *n).round().clip(min=0, max=n).astype(np.int))
        self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self.d2 = (self.d2 / 2 * n).round().clip(min=0, max=n).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.cent = list(np.array(self.cent) / n * 2 - 1.0)
        self.d1 = self.d1 / n * 2
        self.d2 = self.d2 / n * 2

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        chamfer_vec = vec[-1]
        assert chamfer_vec[0] == CHAMFER_IDX
        cent = chamfer_vec[-N_ARGS_FILL_PARAM:-2]
        d1 = chamfer_vec[-2]
        d2 = chamfer_vec[-1]
        res = Chamfer(cent, d1, d2)
        if is_numerical:
            res.denumericalize(n)
        return res

# class Sweep():
#     def __init__(self, cent, d1, d2, is_sym=True):
#         self.cent = cent
#         # for cent in cents:
#         #     self.cents.append([i * 1000 for i in cent])
#         self.d1 = d1
#         self.d2 = d2
#         self.is_sym = is_sym

#     @staticmethod
#     def from_dict(all_stat, fillet_id):
#         fillet_entity = all_stat["entities"][fillet_id]
#         cents = fillet_entity['Edges']
#         d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
#         d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
#         is_sym = fillet_entity['Distance Paramters']['isSymmetric']
#         return [Fillet(1000*cent, d1, d2, is_sym) for cent in cents]

#     def __str__(self):
#         s =  "Fillet Edges: \n"
#         for ep in self.cents:
#             s += "x: {}, y: {}, z:{}\n".format(round(ep[0], 8), round(ep[1], 8), round(ep[2], 8))
#         s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
#         return s

#     def to_vector(self):
#         fillet_vec = np.array(
#             [FILLET_IDX, *[0] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
#         return fillet_vec

#     def transform(self, translation, scale):
#         self.d1 *= scale
#         self.cent = (self + translation) * scale

#     def numericalize(self, n=256):
#         """quantize the representation."""
#         self.cent = ((self.cent + 1.0) / 2 *n).round().clip(min=0, max=n - 1).astype(np.int)
#         self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n - 1).astype(np.int)

#     def denumericalize(self, n=256):
#         """de-quantize the representation."""
#         self.cent = self.cent / n * 2 - 1.0
#         self.d1 = self.d1 / n * 2

#     @staticmethod
#     def from_vector(vec, is_numerical=False, n=256):
#         assert vec[0] == FILLET_IDX
#         cent = vec[-N_ARGS_FILL_PARAM:-1]
#         d = vec[-1]
#         res =  Fillet(cent, d, d)
#         if is_numerical:
#             res.denumericalize(n)
#         return res

# class LinearPattern():
#     def __init__(self, cent, d1, d2, is_sym=True):
#         self.cent = cent
#         # for cent in cents:
#         #     self.cents.append([i * 1000 for i in cent])
#         self.d1 = d1
#         self.d2 = d2
#         self.is_sym = is_sym

#     @staticmethod
#     def from_dict(all_stat, fillet_id):
#         fillet_entity = all_stat["entities"][fillet_id]
#         cents = fillet_entity['Edges']
#         d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
#         d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
#         is_sym = fillet_entity['Distance Paramters']['isSymmetric']
#         return [Fillet(1000*cent, d1, d2, is_sym) for cent in cents]

#     def __str__(self):
#         s =  "Fillet Edges: \n"
#         for ep in self.cents:
#             s += "x: {}, y: {}, z:{}\n".format(round(ep[0], 8), round(ep[1], 8), round(ep[2], 8))
#         s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
#         return s

#     def to_vector(self):
#         fillet_vec = np.array(
#             [FILLET_IDX, *[0] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
#         return fillet_vec

#     def transform(self, translation, scale):
#         self.d1 *= scale
#         self.cent = (self + translation) * scale

#     def numericalize(self, n=256):
#         """quantize the representation."""
#         self.cent = ((self.cent + 1.0) / 2 *n).round().clip(min=0, max=n - 1).astype(np.int)
#         self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n - 1).astype(np.int)

#     def denumericalize(self, n=256):
#         """de-quantize the representation."""
#         self.cent = self.cent / n * 2 - 1.0
#         self.d1 = self.d1 / n * 2

#     @staticmethod
#     def from_vector(vec, is_numerical=False, n=256):
#         assert vec[0] == FILLET_IDX
#         cent = vec[-N_ARGS_FILL_PARAM:-1]
#         d = vec[-1]
#         res =  Fillet(cent, d, d)
#         if is_numerical:
#             res.denumericalize(n)
#         return res

# class CirclePattern():
#     def __init__(self, cent, d1, d2, is_sym=True):
#         self.cent = cent
#         # for cent in cents:
#         #     self.cents.append([i * 1000 for i in cent])
#         self.d1 = d1
#         self.d2 = d2
#         self.is_sym = is_sym

#     @staticmethod
#     def from_dict(all_stat, fillet_id):
#         fillet_entity = all_stat["entities"][fillet_id]
#         cents = fillet_entity['Edges']
#         d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
#         d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
#         is_sym = fillet_entity['Distance Paramters']['isSymmetric']
#         return [Fillet(1000*cent, d1, d2, is_sym) for cent in cents]

#     def __str__(self):
#         s =  "Fillet Edges: \n"
#         for ep in self.cents:
#             s += "x: {}, y: {}, z:{}\n".format(round(ep[0], 8), round(ep[1], 8), round(ep[2], 8))
#         s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
#         return s

#     def to_vector(self):
#         fillet_vec = np.array(
#             [FILLET_IDX, *[0] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
#         return fillet_vec

#     def transform(self, translation, scale):
#         self.d1 *= scale
#         self.cent = (self + translation) * scale

#     def numericalize(self, n=256):
#         """quantize the representation."""
#         self.cent = ((self.cent + 1.0) / 2 *n).round().clip(min=0, max=n - 1).astype(np.int)
#         self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n - 1).astype(np.int)

#     def denumericalize(self, n=256):
#         """de-quantize the representation."""
#         self.cent = self.cent / n * 2 - 1.0
#         self.d1 = self.d1 / n * 2

#     @staticmethod
#     def from_vector(vec, is_numerical=False, n=256):
#         assert vec[0] == FILLET_IDX
#         cent = vec[-N_ARGS_FILL_PARAM:-1]
#         d = vec[-1]
#         res =  Fillet(cent, d, d)
#         if is_numerical:
#             res.denumericalize(n)
#         return res

# class MirrorPattern():
#     def __init__(self, cent, d1, d2, is_sym=True):
#         self.cent = cent
#         # for cent in cents:
#         #     self.cents.append([i * 1000 for i in cent])
#         self.d1 = d1
#         self.d2 = d2
#         self.is_sym = is_sym

#     @staticmethod
#     def from_dict(all_stat, fillet_id):
#         fillet_entity = all_stat["entities"][fillet_id]
#         cents = fillet_entity['Edges']
#         d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
#         d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
#         is_sym = fillet_entity['Distance Paramters']['isSymmetric']
#         return [Fillet(1000*cent, d1, d2, is_sym) for cent in cents]

#     def __str__(self):
#         s =  "Fillet Edges: \n"
#         for ep in self.cents:
#             s += "x: {}, y: {}, z:{}\n".format(round(ep[0], 8), round(ep[1], 8), round(ep[2], 8))
#         s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
#         return s

#     def to_vector(self):
#         fillet_vec = np.array(
#             [FILLET_IDX, *[0] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
#         return fillet_vec

#     def transform(self, translation, scale):
#         self.d1 *= scale
#         self.cent = (self + translation) * scale

#     def numericalize(self, n=256):
#         """quantize the representation."""
#         self.cent = ((self.cent + 1.0) / 2 *n).round().clip(min=0, max=n - 1).astype(np.int)
#         self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n - 1).astype(np.int)

#     def denumericalize(self, n=256):
#         """de-quantize the representation."""
#         self.cent = self.cent / n * 2 - 1.0
#         self.d1 = self.d1 / n * 2

#     @staticmethod
#     def from_vector(vec, is_numerical=False, n=256):
#         assert vec[0] == FILLET_IDX
#         cent = vec[-N_ARGS_FILL_PARAM:-1]
#         d = vec[-1]
#         res =  Fillet(cent, d, d)
#         if is_numerical:
#             res.denumericalize(n)
#         return res