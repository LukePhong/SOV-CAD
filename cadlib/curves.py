from copy import deepcopy
import numpy as np
import matplotlib.lines as lines
import matplotlib.patches as patches
from .math_utils import rads_to_degs, angle_from_vector_to_x
from .macro import *
import matplotlib.transforms as transforms


# FIXME: these two functions can be treated as static method
def construct_curve_from_dict(stat):
    if stat['type'] == "Line3D":
        return Line.from_dict(stat)
    elif stat['type'] == "Circle3D":
        return Circle.from_dict(stat)
    elif stat['type'] == "Arc3D":
        return Arc.from_dict(stat)
    else:
        raise NotImplementedError("curve type not supported yet: {}".format(stat['type']))


def construct_curve_from_vector(vec, start_point, is_numerical=True):
    type = vec[0]
    if type == LINE_IDX:
        return Line.from_vector(vec, start_point, is_numerical=is_numerical)
    elif type == CIRCLE_IDX:
        return Circle.from_vector(vec, start_point, is_numerical=is_numerical)
    elif type == ARC_IDX:
        res = Arc.from_vector(vec, start_point, is_numerical=is_numerical)
        if res is None: # for visualization purpose, replace illed arc with line
            return Line.from_vector(vec, start_point, is_numerical=is_numerical)
        return res
    else:
        raise NotImplementedError("curve type not supported yet: command idx {}".format(vec[0]))


#######################  base  #######################
class CurveBase(object):
    """Base class for curve. All types of curves shall inherit from this."""
    def __init__(self):
        pass

    @staticmethod
    def from_dict(stat):
        """construct curve from json data"""
        raise NotImplementedError

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        """construct curve from vector representation"""
        raise NotImplementedError

    @property
    def bbox(self):
        """compute bounding box of the curve"""
        raise NotImplementedError

    def direction(self, from_start=True):
        """return a vector indicating the curve direction"""
        raise NotImplementedError

    def transform(self, translate, scale):
        """linear transformation"""
        raise NotImplementedError

    def flip(self, axis):
        """flip the curve about axis"""
        raise NotImplementedError

    def reverse(self):
        """reverse the curve direction"""
        raise NotImplementedError

    def numericalize(self, n=256):
        """quantize curve parameters into integers"""
        raise NotImplementedError

    def to_vector(self):
        """represent curve using a vector. see macro.py"""
        raise NotImplementedError

    def draw(self, ax, color, line_width=1):
        """draw the curve using matplotlib"""
        raise NotImplementedError

    def sample_points(self, n=32):
        """uniformly sample points from the curve"""
        raise NotImplementedError


####################### curves #######################
class Line(CurveBase):
    def __init__(self, start_point, end_point):
        super(Line, self).__init__()
        self.start_point = start_point
        self.end_point = end_point

    def __str__(self):
        return "Line: start({}), end({})".format(self.start_point.round(4), self.end_point.round(4))

    @staticmethod
    def from_dict(stat):
        assert stat['type'] == "Line3D"
        start_point = 1000 * np.array([stat['start_point']['x'],
                                stat['start_point']['y']])
        end_point = 1000 * np.array([stat['end_point']['x'],
                              stat['end_point']['y']])
        return Line(start_point, end_point)

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        return Line(start_point, vec[1:3])

    @property
    def bbox(self):
        points = np.stack([self.start_point, self.end_point], axis=0)
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)

    def direction(self, from_start=True):
        return self.end_point - self.start_point

    def transform(self, translate, scale):
        self.start_point = (self.start_point + translate) * scale
        self.end_point = (self.end_point + translate) * scale

    def flip(self, axis):
        if axis == 'x':
            self.start_point[1], self.end_point[1] = -self.start_point[1], -self.end_point[1]
        elif axis == 'y':
            self.start_point[0], self.end_point[0] = -self.start_point[0], -self.end_point[0]
        elif axis == 'xy':
            self.start_point = self.start_point * -1
            self.end_point = self.end_point * -1
        else:
            raise ValueError("axis = {}".format(axis))

    def reverse(self):
        self.start_point, self.end_point = self.end_point, self.start_point

    def numericalize(self, n=256):
        self.start_point = self.start_point.round().clip(min=0, max=n).astype(np.int)
        self.end_point = self.end_point.round().clip(min=0, max=n).astype(np.int)

    def to_vector(self):
        vec = [LINE_IDX, self.end_point[0], self.end_point[1]]
        return np.array(vec + [PAD_VAL] * (1 + N_ARGS - len(vec)))

    def draw(self, ax, color, line_width=1):
        xdata = [self.start_point[0], self.end_point[0]]
        ydata = [self.start_point[1], self.end_point[1]]
        l1 = lines.Line2D(xdata, ydata, lw=line_width, color=color, axes=ax)
        ax.add_line(l1)
        ax.plot(self.start_point[0], self.start_point[1], 'o', color=color, markersize=max(3, line_width*2))
        # ax.plot(self.end_point[0], self.end_point[1], 'o')
        
    def draw_image(self, fig, ax, color, img=[], line_width=1):
        self.draw(ax, color, line_width)
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        img.append(deepcopy(img_array))

    def sample_points(self, n=32):
        return np.linspace(self.start_point, self.end_point, num=n)


class Arc(CurveBase):
    def __init__(self, start_point, end_point, center, radius,
                 normal=None, start_angle=None, end_angle=None, ref_vec=None):
        super(Arc, self).__init__()
        self.start_point = start_point
        self.end_point = end_point
        self.center = center
        self.radius = radius
        self.normal = normal
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.ref_vec = ref_vec
        self.mid_point = self.get_mid_point()

    def __str__(self):
        return "Arc: start({}), end({}), mid({})".format(self.start_point.round(4), self.end_point.round(4),
                                                         self.mid_point.round(4))

    @staticmethod
    def from_dict(stat):
        assert stat['type'] == "Arc3D"
        start_point = 1000 * np.array([stat['start_point']['x'],
                                stat['start_point']['y']])
        end_point = 1000 * np.array([stat['end_point']['x'],
                              stat['end_point']['y']])
        center = 1000 * np.array([stat['center_point']['x'],
                           stat['center_point']['y']])
        radius = 1000 * stat['radius']
        normal_key = "normal_vec" if "normal_vec" in stat else "normal"
        normal = np.array([stat[normal_key]['x'],
                           stat[normal_key]['y'],
                           stat[normal_key]['z']])

        if normal_key == "normal":
            # orignal code
            start_angle = stat['start_angle']
            end_angle = stat['end_angle']
            ref_vec = np.array([stat['reference_vector']['x'],
                                stat['reference_vector']['y']])
            
            # normalize ref_vec
            ref_vec = ref_vec / np.linalg.norm(ref_vec)

            return Arc(start_point, end_point, center, radius, normal, 
                   start_angle, end_angle, ref_vec)

        # Calculate start_angle, end_angle, and ref_vec
        epsilon = 1e-7  # Small constant for floating point comparisons

        if radius < epsilon: # Degenerate case: tiny radius
            # Default to a zero-length arc, with ref_vec along the positive x-axis.
            ref_vec_calc = np.array([1.0, 0.0])
            start_angle_calc = 0.0
            end_angle_calc = 0.0
        else:
            v_start_to_center = start_point - center
            norm_v_start = np.linalg.norm(v_start_to_center)

            if norm_v_start < epsilon:
                # Start point is at the center. This is problematic if radius > 0.
                # Default ref_vec to x-axis. Arc will be ill-defined but code won't crash.
                v_start_normalized = np.array([1.0, 0.0])
            else:
                v_start_normalized = v_start_to_center / norm_v_start
            
            ref_vec_calc = v_start_normalized
            start_angle_calc = 0.0 # Arc definition: starts along its reference vector

            # Check for full circle case (start_point is very close to end_point)
            if np.linalg.norm(start_point - end_point) < epsilon: # Compare scaled points
                end_angle_calc = 0.0 # Default to zero sweep
                if normal[2] > epsilon:  # CCW full circle
                    end_angle_calc = 2 * np.pi
                elif normal[2] < -epsilon:  # CW full circle
                    end_angle_calc = -2 * np.pi
                # If normal[2] is near zero, ambiguous, treat as zero-length.
            else:
                # Standard arc with distinct start and end points
                v_end_to_center = end_point - center
                norm_v_end = np.linalg.norm(v_end_to_center)

                if norm_v_end < epsilon:
                    # End point is at the center. Arc is ill-defined.
                    # Treat as zero sweep by using start vector's direction.
                    v_end_normalized = v_start_normalized
                else:
                    v_end_normalized = v_end_to_center / norm_v_end
                
                # Calculate global angles of NORMALIZED vectors w.r.t. X-axis
                angle_s_global = angle_from_vector_to_x(v_start_normalized)
                angle_e_global = angle_from_vector_to_x(v_end_normalized)

                # Calculate sweep angle
                end_angle_calc = angle_e_global - angle_s_global

                # Adjust sweep angle based on normal_vec.z to ensure correct winding
                # and handle reflex angles.
                if normal[2] > epsilon:  # Expect CCW sweep
                    if end_angle_calc < -epsilon: # If raw sweep is CW, add 2*pi for CCW reflex
                        end_angle_calc += 2 * np.pi
                elif normal[2] < -epsilon:  # Expect CW sweep
                    if end_angle_calc > epsilon: # If raw sweep is CCW, subtract 2*pi for CW reflex
                        end_angle_calc -= 2 * np.pi
                # If abs(normal[2]) < epsilon, end_angle_calc remains the shortest sweep.
        
        return Arc(start_point, end_point, center, radius, normal, 
                   start_angle_calc, end_angle_calc, ref_vec_calc)

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        # print(vec)
        end_point = vec[1:3]
        sweep_angle = vec[3] / 255 * 2 * np.pi# if is_numerical else vec[3]
        clock_sign = vec[4]
        s2e_vec = end_point - start_point
        if np.linalg.norm(s2e_vec) == 0:
            return None
        radius = (np.linalg.norm(s2e_vec) / 2) / np.sin(sweep_angle / 2)
        s2e_mid = (start_point + end_point) / 2
        vertical = np.cross(s2e_vec, [0, 0, 1])[:2]
        vertical = vertical / np.linalg.norm(vertical)
        if clock_sign == 0:
            vertical = -vertical
        center_point = s2e_mid - vertical * (radius * np.cos(sweep_angle / 2))

        start_angle = 0
        end_angle = sweep_angle
        if clock_sign == 0:
            ref_vec = end_point - center_point
        else:
            ref_vec = start_point - center_point
        ref_vec = ref_vec / np.linalg.norm(ref_vec)

        return Arc(start_point, end_point, center_point, radius,
                   start_angle=start_angle, end_angle=end_angle, ref_vec=ref_vec)

    def get_angles_counterclockwise(self, eps=1e-8):
        c2s_vec = (self.start_point - self.center) / (np.linalg.norm(self.start_point - self.center) + eps)
        c2m_vec = (self.mid_point - self.center) / (np.linalg.norm(self.mid_point - self.center) + eps)
        c2e_vec = (self.end_point - self.center) / (np.linalg.norm(self.end_point - self.center) + eps)
        angle_s, angle_m, angle_e = angle_from_vector_to_x(c2s_vec), angle_from_vector_to_x(c2m_vec), \
                                    angle_from_vector_to_x(c2e_vec)
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s
        return angle_s, angle_e

    @property
    def bbox(self):
        points = [self.start_point, self.end_point]
        angle_s, angle_e = self.get_angles_counterclockwise()
        if angle_s < 0 < angle_e:
            points.append(np.array([self.center[0] + self.radius, self.center[1]]))
        if angle_s < np.pi / 2 < angle_e or angle_s < -np.pi / 2 * 3 < angle_e:
            points.append(np.array([self.center[0], self.center[1] + self.radius]))
        if angle_s < np.pi < angle_e or angle_s < -np.pi < angle_e:
            points.append(np.array([self.center[0] - self.radius, self.center[1]]))
        if angle_s < np.pi / 2 * 3 < angle_e or angle_s < -np.pi/2 < angle_e:
            points.append(np.array([self.center[0], self.center[1] - self.radius]))
        points = np.stack(points, axis=0)
        return np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0)

    def direction(self, from_start=True):
        if from_start:
            return self.mid_point - self.start_point
        else:
            return self.end_point - self.mid_point

    @property
    def clock_sign(self):
        """get a boolean sign indicating whether the arc is on top of s->e """
        s2e = self.end_point - self.start_point
        s2m = self.mid_point - self.start_point
        sign = np.cross(s2m, s2e) >= 0 # counter-clockwise
        return sign

    def get_mid_point(self):
        mid_angle = (self.start_angle + self.end_angle) / 2
        rot_mat = np.array([[np.cos(mid_angle), -np.sin(mid_angle)],
                            [np.sin(mid_angle), np.cos(mid_angle)]])
        mid_vec = rot_mat @ self.ref_vec
        return self.center + mid_vec * self.radius

    def transform(self, translate, scale):
        self.start_point = (self.start_point + translate) * scale
        self.mid_point = (self.mid_point + translate) * scale
        self.end_point = (self.end_point + translate) * scale
        self.center = (self.center + translate) * scale
        if isinstance(scale * 1.0, float):
            self.radius = abs(self.radius * scale)

    def flip(self, axis):
        if axis == 'x':
            self.transform(0, np.array([1, -1]))
            new_ref_vec_angle = angle_from_vector_to_x(self.ref_vec) + self.end_angle - self.start_angle
            self.ref_vec = np.array([np.cos(new_ref_vec_angle), -np.sin(new_ref_vec_angle)])
        elif axis == 'y':
            self.transform(0, np.array([-1, 1]))
            new_ref_vec_angle = angle_from_vector_to_x(self.ref_vec) + self.end_angle - self.start_angle
            self.ref_vec = np.array([-np.cos(new_ref_vec_angle), np.sin(new_ref_vec_angle)])
        elif axis == 'xy':
            self.transform(0, -1)
            self.ref_vec = self.ref_vec * -1
        else:
            raise ValueError("axis = {}".format(axis))

    def reverse(self):
        self.start_point, self.end_point = self.end_point, self.start_point

    def numericalize(self, n=256):
        self.start_point = self.start_point.round().clip(min=0, max=n).astype(np.int)
        self.mid_point = self.mid_point.round().clip(min=0, max=n).astype(np.int)
        self.end_point = self.end_point.round().clip(min=0, max=n).astype(np.int)
        self.center = self.center.round().clip(min=0, max=n).astype(np.int)
        tmp = np.array([self.start_angle, self.end_angle])
        self.start_angle, self.end_angle = (tmp / (2 * np.pi) * n).round().clip(
                                            min=0, max=n).astype(np.int)

    def to_vector(self):
        sweep_angle = max(abs(self.start_angle - self.end_angle), 1)
        return np.array([ARC_IDX, self.end_point[0], self.end_point[1], sweep_angle, int(self.clock_sign), PAD_VAL,
                         *[PAD_VAL] * (N_ARGS - N_ARGS_SKETCH)])

    def draw(self, ax, color, line_width=1):
        ref_vec_angle = rads_to_degs(angle_from_vector_to_x(self.ref_vec))
        start_angle = rads_to_degs(self.start_angle)
        end_angle = rads_to_degs(self.end_angle)
        
        # # Calculate the actual arc extent to determine appropriate dimensions
        # # Calculate the bounding box based on the arc points
        # points = np.vstack([self.start_point, self.mid_point, self.end_point])
        # min_xy = np.min(points, axis=0)
        # max_xy = np.max(points, axis=0)
        # bbox_width = max_xy[0] - min_xy[0]
        # bbox_height = max_xy[1] - min_xy[1]
            
        # if ref_vec_angle > 180:
        width = 2 * self.radius
        height = 2 * self.radius
        # else:
        #     width = bbox_width
        #     height = bbox_height
        
        ap = patches.Arc(
            (self.center[0], self.center[1]),
            width,
            height,
            angle=ref_vec_angle,
            theta1=start_angle,
            theta2=end_angle,
            lw=line_width,
            color=color,
        )
        
        ax.add_patch(ap)
        ax.plot(self.start_point[0], self.start_point[1], 'o', color=color, markersize=max(3, line_width*2))
        ax.plot(self.mid_point[0], self.mid_point[1], 'o', color=color, markersize=max(3, line_width*2))

    def draw_image(self, fig, ax, color, img=[], line_width=1):
        self.draw(ax, color, line_width)
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        img.append(deepcopy(img_array))

    def sample_points(self, n=32):
        c2s_vec = (self.start_point - self.center) / np.linalg.norm(self.start_point - self.center)
        c2m_vec = (self.mid_point - self.center) / np.linalg.norm(self.mid_point - self.center)
        c2e_vec = (self.end_point - self.center) / np.linalg.norm(self.end_point - self.center)
        angle_s, angle_m, angle_e = angle_from_vector_to_x(c2s_vec), angle_from_vector_to_x(c2m_vec), \
                                    angle_from_vector_to_x(c2e_vec)
        angle_s, angle_e = min(angle_s, angle_e), max(angle_s, angle_e)
        if not angle_s < angle_m < angle_e:
            angle_s, angle_e = angle_e - np.pi * 2, angle_s

        angles = np.linspace(angle_s, angle_e, num=n)
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.radius + self.center[np.newaxis]
        return points


class Circle(CurveBase):
    def __init__(self, center, radius, normal=None):
        super(Circle, self).__init__()
        self.center = center
        self.radius = radius
        self.normal = normal

    def __str__(self):
        return "Circle: center({}), radius({})".format(self.center.round(4), round(self.radius, 4))

    @staticmethod
    def from_dict(stat):
        assert stat['type'] == "Circle3D"
        center = 1000 * np.array([stat['center_point']['x'],
                           stat['center_point']['y']])
        radius = 1000 * stat['radius']
        # normal = np.array([stat['normal']['x'],
        #                    stat['normal']['y'],
        #                    stat['normal']['z']])
        return Circle(center, radius)

    @staticmethod
    def from_vector(vec, start_point=None, is_numerical=True):
        return Circle(vec[1:3], vec[5])

    @property
    def bbox(self):
        return np.stack([self.center - self.radius, self.center + self.radius], axis=0)

    def direction(self, from_start=True):
        return self.center - self.start_point

    @property
    def start_point(self):
        return np.array([self.center[0] - self.radius, self.center[1]])

    @property
    def end_point(self):
        return np.array([self.center[0] + self.radius, self.center[1]])

    def transform(self, translate, scale):
        self.center = (self.center + translate) * scale
        self.radius = self.radius * scale

    def flip(self, axis):
        if axis == 'x':
            self.center[1] = -self.center[1]
        elif axis == 'y':
            self.center[0] = -self.center[0]
        elif axis == 'xy':
            self.center = self.center * -1
        else:
            raise ValueError("axis = {}".format(axis))

    def reverse(self):
        pass

    def numericalize(self, n=256):
        self.center = self.center.round().clip(min=0, max=n).astype(np.int)
        self.radius = np.round(self.radius).clip(min=1, max=n).astype(np.int)

    def to_vector(self):
        vec = [CIRCLE_IDX, self.center[0], self.center[1], PAD_VAL, PAD_VAL, self.radius]
        return np.array(vec + [PAD_VAL] * (1 + N_ARGS - len(vec)))

    def draw(self, ax, color, line_width=1):
        ap = patches.Circle((self.center[0], self.center[1]), self.radius,
                            lw=line_width, fill=None, color=color)
        ax.add_patch(ap)
        ax.plot(self.center[0], self.center[1], 'o', color=color, markersize=max(3, line_width*2))

    def draw_image(self, fig, ax, color, img=[], line_width=1):
        self.draw(ax, color, line_width)
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        img.append(deepcopy(img_array))

    def sample_points(self, n=32):
        angles = np.linspace(0, np.pi * 2, num=n, endpoint=False)
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.radius + self.center[np.newaxis]
        return points
