import math

import cv2
import numpy as np
from dtld_parsing.calibration import CalibrationData

from typing import Tuple

__author__ = "Andreas Fregin, Julian Mueller and Klaus Dietmayer"
__maintainer__ = "Julian Mueller"
__email__ = "julian.mu.mueller@daimler.com"


class ThreeDPosition(object):
    """
    Three dimensional position with respect to a defined frame_id.
    """

    def __init__(self, x: float, y: float, z: float, frame_id: str = "stereo_left"):
        self._x = x
        self._y = y
        self._z = z
        self._frame_id = frame_id

    def set_pos(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z

    def move_pos(self, x: float, y: float, z: float):
        self._x += x
        self._y += y
        self._z += z

    def get_pos(self) -> Tuple[float, float, float]:
        return self._x, self._y, self._z


class ThreeDimensionalPosition(object):
    def __init__(
        self,
        calibration_left: CalibrationData,
        calibration_right: CalibrationData,
        binning_x: int = 0,
        binning_y: int = 0,
        roi_offset_x: int = 0,
        roi_offset_y: int = 0,
    ):
        """
        Class determining the 3D position of objects from disparity images.

        Args:
            calibration_left(CalibrationData): calibration for left camera
            calibration_right(CalibrationData): calibration for right camera
            binning_x(int): binning between original camera and disparity image in x direction
            binning_y(int): binning between original camera and disparity image in y direction
            roi_offset_x(int): RoI offset in x
            roi_offset_y(int): RoI offset in y
        """
        self._calibration_left = calibration_left
        self._calibration_right = calibration_right
        self._binning_x = binning_x
        self._binning_y = binning_y
        self._roi_offset_x = roi_offset_x
        self._roi_offset_y = roi_offset_y

    def unrectify_rectangle(self, x: int, y: int, width: int, height: int):
        """
        Converts rectified to unrectified coordinates using calibration matrices.

        Args:
            x(int): upper left corner of bbox
            y(int): upper left corner of bbox
            width(int): width of bbox
            height(int): height of bbox

        Returns:
            x, y, width, height in unrectified coordinates
        """
        # not rectified coordinates
        pt_distorted = np.array([[float(x), float(y)], [float(x + width), float(y + height),],])
        pt_distorted = pt_distorted[:, np.newaxis, :]

        # rectify points
        pt_undistorted = cv2.undistortPoints(
            pt_distorted,
            self._calibration_left.intrinsic_calibration.intrinsic_matrix,
            self._calibration_left.distortion_calibration.distortion_matrix,
            R=self._calibration_left.rectification_matrix.rectification_matrix,
            P=self._calibration_left.projection_matrix.projection_matrix,
        )

        # get new coords
        x_out = pt_undistorted[0][0][0]
        y_out = pt_undistorted[0][0][1]
        w_out = pt_undistorted[1][0][0] - pt_undistorted[0][0][0]
        h_out = pt_undistorted[1][0][1] - pt_undistorted[0][0][1]

        # binning in x and y (camera images were binned before
        # disparity calculation)
        return (
            int(round(x_out / float(self._binning_x))),
            int(round(y_out / float(self._binning_y))),
            int(round(w_out / float(self._binning_x))),
            int(round(h_out / float(self._binning_y))),
        )

    def determine_disparity(self, x: int, y: int, width: int, height: int, disparity_image: np.ndarray) -> float:
        """
        Calculates disparity from unrectified coordinates using calibration matrices and disparity image input.

        Args:
            x(int): upper left corner of bbox
            y(int): upper left corner of bbox
            width(int): width of bbox
            height(int): height of bbox
            disparity_image(np.ndarray): disparity image

        Returns:
            float: median disparity in RoI
        """
        disparity_crop = disparity_image[y : y + height, x : x + width]
        # image = cv2.rectangle(
        #    disparity_image, (int(x), int(y)), (int(x) + int(width), int(y) + int(height)), (255, 255, 255), 1,
        # )
        # cv2.imwrite("/home/muelju3/disp.png", image)
        return np.nanmedian(disparity_crop)

    def determine_three_dimensional_position(
        self, x: int, y: int, width: int, height: int, disparity_image: np.ndarray
    ) -> ThreeDPosition:
        """
        Calculates 3d position from rectified coordinates using calibration matrices and disparity image input.

        Args:
            x(int): upper left corner of bbox
            y(int): upper left corner of bbox
            width(int): width of bbox
            height(int): weight of bbox
            disparity_image(np.ndarray): disparity image

        Returns:
            ThreeDPosition: ThreeDPosition
        """
        x_u, y_u, width_u, height_u = self.unrectify_rectangle(x=x, y=y, width=width, height=height)

        disparity = self.determine_disparity(
            x=x_u - int(round(self._roi_offset_x / self._binning_x)),
            y=y_u - int(round(self._roi_offset_y / self._binning_y)),
            width=width_u,
            height=height_u,
            disparity_image=disparity_image,
        )

        # all values inside bbox are nan --> no depth
        if disparity == 0.0 or math.isnan(disparity):
            return ThreeDPosition(x=-1.0, y=-1.0, z=-1.0, frame_id="stereo_left")

        return self.twod_point_to_threed_from_disparity(x=x + width / 2.0, y=y + height / 2.0, disparity=disparity)

    def twod_point_to_threed_from_disparity(self, x, y, disparity):

        # get calibration values
        left_fx = self._calibration_left.intrinsic_calibration.fx
        left_fy = self._calibration_left.intrinsic_calibration.fy
        left_cx = self._calibration_left.intrinsic_calibration.cx
        left_cy = self._calibration_left.intrinsic_calibration.cy
        tx = -1.0 * self._calibration_right.projection_matrix.baseline

        # determine 3d pos
        x_world = left_fy * tx * x - left_fy * left_cx * tx
        y_world = left_fx * tx * y - left_fx * left_cy * tx
        z_world = left_fx * left_fy * tx

        # normalize
        w = -1.0 * self._binning_x * left_fy * disparity

        return ThreeDPosition(x=x_world / w, y=y_world / w, z=z_world / w, frame_id="stereo_left")

    def twod_point_to_threed_from_depth(self, x: int, y: int, depth: float) -> float:

        disparity = self.depth_to_disparity(depth)
        return self.twod_point_to_threed_from_disparity(x, y, disparity)

    def disparity_to_depth(self, disparity: float) -> float:
        """
        Converts disparity to depth.

        Args:
            disparity(float): Disparity in pixels

        Returns:
            float: depth value in meters
        """
        tx = -1.0 * self._calibration_right.projection_matrix.tx
        return tx / (disparity * self._binning_x)

    def depth_to_disparity(self, depth: float) -> float:
        """
        Converts depth to disparity.

        Args:
            depth(float): Depth in meters

        Returns:
            float: disparity in meters
        """
        tx = -1.0 * self._calibration_right.projection_matrix.tx
        return tx / (depth * self._binning_x)

    def twod_from_threed(self, x: float, y: float, z: float):
        """
        Calculates hypothesis size in pixels based on depth of object.

        Args:
            x(float): 3D position x coordinate
            y(float): 3D position z coordinate
            z(float): 3D position y coordinate

        Returns:
            int, int: 2d pos
        """
        # translation = depth
        t_vec = np.array([0.0, 0.0, 0.0])
        r_vec = np.array([0.0, 0.0, 0.0])

        # world corner points of object (float object assumption)
        world_points = np.array([[x, y, z],])

        # project world points on image plane
        image_points = cv2.projectPoints(
            world_points,
            r_vec,
            t_vec,
            self._calibration_left.intrinsic_calibration.intrinsic_matrix,
            distCoeffs=self._calibration_left.distortion_calibration.distortion_matrix,
        )[0].tolist()

        # determine box width and height
        return image_points[0][0][0], image_points[0][0][1]
