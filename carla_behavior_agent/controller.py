# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
import carla
from misc import get_speed
import sys

from math import atan2, atan, hypot

class VehicleController():
    """
    VehicleController is the combination of longitudinal PID controller
    and a Stanley as lateral controller to perform the low level control 
    a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3,
                 max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral Stanley controller
        using the following semantics:
            K_V -- Gain term
            K_S -- Stability term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = StanleyLateralController(self._vehicle, offset, **args_lateral)


    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """

        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step()
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_controller(self, args_lateral):
        """Changes the parameters of the StanleyLateralController"""
        self._lat_controller.change_parameters(**args_lateral)
    
    def setWaypoints(self, waypoints):
        self._lat_controller.setWaypoints(waypoints)


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

class StanleyLateralController():
    """
    StanleyLateralController implements lateral control using a Stanley.
    """

    def __init__(self, vehicle, offset=0, lookahead_distance=1.0, K_V=1.0, K_S=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param lookahead_distance: Distance to lookahead
            :param K_V: Proportional term
            :param K_S: Differential term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._kv = K_V
        self._ks = K_S
        self._dt = dt
        self._wps = None
        self._lookahead_distance = lookahead_distance
        self._offset = offset

    def run_step(self):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._stanley_control(self._vehicle.get_transform())
    
    def _get_lookahead_index(self, ego_loc, lookahead_distance):
        min_idx       = 0
        min_dist      = float("inf")
        for i in range(len(self._wps)):
            dist = np.linalg.norm(np.array([
                    self._wps[i][0].transform.location.x - ego_loc.x,
                    self._wps[i][0].transform.location.y - ego_loc.y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        total_dist = min_dist
        lookahead_idx = min_idx
        for i in range(min_idx + 1, len(self._wps)):
            if total_dist >= lookahead_distance:
                break
            total_dist += np.linalg.norm(np.array([
                    self._wps[i][0].transform.location.x - self._wps[i-1][0].transform.location.x,
                    self._wps[i][0].transform.location.y - self._wps[i-1][0].transform.location.y]))
            lookahead_idx = i
        return lookahead_idx
    
    def _stanley_control(self, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the Stanley equations

            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get ego vehicle observations
        ego_loc = vehicle_transform.location
        speed_estimate = get_speed(self._vehicle)
        observed_heading = np.deg2rad(vehicle_transform.rotation.yaw)
        observed_x = ego_loc.x
        observed_y = ego_loc.y
        
        # Get Target Waypoint
        ce_idx = self._get_lookahead_index(ego_loc,self._lookahead_distance)
        desired_x = self._wps[ce_idx][0].transform.location.x
        desired_y = self._wps[ce_idx][0].transform.location.y
        
        # Get Target Heading
        if ce_idx < len(self._wps)-1:
            desired_heading_x = self._wps[ce_idx+1][0].transform.location.x - self._wps[ce_idx][0].transform.location.x
            desired_heading_y = self._wps[ce_idx+1][0].transform.location.y - self._wps[ce_idx][0].transform.location.y
        else:
            desired_heading_x = self._wps[ce_idx][0].transform.location.x - self._wps[ce_idx-1][0].transform.location.x
            desired_heading_y = self._wps[ce_idx][0].transform.location.y - self._wps[ce_idx-1][0].transform.location.y
        
        # Trajectory Heading
        desired_heading = atan2(desired_heading_y, desired_heading_x)
        
        # Trajectory Distance
        dd=hypot(desired_heading_x, desired_heading_y)
                        
        # Crosstrack error
        lateral_error = \
              ((observed_x-desired_x)*desired_heading_y -
              (observed_y-desired_y)*desired_heading_x) / (dd + sys.float_info.epsilon) + self._offset

        
        # Heading error
        steering = (desired_heading-observed_heading)
        
        # Normalization to [-pi, pi]
        while (steering<-np.pi):
            steering += 2*np.pi
        while (steering>np.pi):
            steering -= 2*np.pi
            
        steering_error = steering
                
        steering += atan(self._kv * lateral_error /
                               (self._ks + speed_estimate))
        
        # print("Current Heading: ", observed_heading, " - Desired Heading: ", desired_heading)
        # print("Heading error: ", steering_error, "Crosstrack error: ", lateral_error)
        # print("Output: ", steering)
        
        return np.clip(steering, -1.0, 1.0)

    def change_parameters(self, Kv, Ks, dt):
        """Changes the Stanley parameters"""
        self._kv = Kv
        self._ks = Ks
        self._dt = dt
    
    def setWaypoints(self, wps):
        """Sets trajectory to follow and filters spurious points"""
        self._wps = [wps[0]]
        for i in range(1, len(wps) - 1):
            trj_heading_x = wps[i][0].transform.location.x - self._wps[-1][0].transform.location.x
            trj_heading_y = wps[i][0].transform.location.y - self._wps[-1][0].transform.location.y
            
            dd = hypot(trj_heading_x, trj_heading_y)
            if dd > 0:
                self._wps.append(wps[i])
    
    @property
    def offset(self):
        '''
        Get the offset of the vehicle from the center of the lane.
        '''
        return self._offset
    
    @offset.setter
    def offset(self, offset : int):
        '''
        Set the offset of the vehicle from the center of the lane.
        '''
        self._offset = offset
        
class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, offset=0, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        else:
            w_loc = waypoint.transform.location

        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])

        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            _dot = 1
        else:
            _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt