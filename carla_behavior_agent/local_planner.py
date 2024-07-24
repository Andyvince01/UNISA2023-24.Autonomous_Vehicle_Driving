# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

import math
import random
from enum import IntEnum
from collections import deque

import carla
from controller import VehicleController
from misc import get_speed

# ====================================================================
# -- RoadOption Enum ------------------------------------------------
# ====================================================================
class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving 
    from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

# ====================================================================
# -- LocalPlanner ---------------------------------------------------
# ====================================================================
class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a
    trajectory of waypoints that is generated on-the-fly.

    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice,
    unless a given global plan has already been specified.
    """

    # ====================================================================
    # -- LocalPlanner Init ----------------------------------------------
    # ====================================================================
    def __init__(self, vehicle: carla.Actor, opt_dict: dict = {}, map_inst=None):
        """
        Init method for the local planner.
        
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with different parameters:
            dt: time between simulation steps
            target_speed: desired cruise speed in Km/h
            sampling_radius: distance between the waypoints part of the plan
            lateral_control_dict: values of the lateral PID controller
            longitudinal_control_dict: values of the longitudinal PID controller
            max_throttle: maximum throttle applied to the vehicle
            max_brake: maximum brake applied to the vehicle
            max_steering: maximum steering applied to the vehicle
            offset: distance between the route waypoints and the center of the lane
        :param map_inst: carla.Map instance to avoid the expensive call of getting it.
        """
        # Basic parameters
        self._vehicle : carla.Actor = vehicle
        self._world : carla.World = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()

        # Controller parameters
        self._vehicle_controller = None
        self.target_waypoint = None
        self.target_road_option = None

        # Waypoints queue
        self._waypoints_queue = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False

        # Base parameters
        if not isinstance(opt_dict, dict):
            raise ValueError("opt_dict must be a dictionary!")
                
        self._args_lateral_dict = opt_dict.get('lateral_control_dict', {'K_S': 0.0, 'K_V': 0.0, 'dt': 0.0})                         # Lateral PID controller values
        self._args_longitudinal_dict = opt_dict.get('longitudinal_control_dict', {'K_D': 0.0, 'K_I': 0.0, 'K_P': 0.0, 'dt': 0.0})   # Longitudinal PID controller values
        self._base_min_distance = opt_dict.get('base_min_distance', 3.0)                                                            # Minimum distance to keep from the vehicle ahead
        self._distance_ratio = opt_dict.get('distance_ratio', 0.5)                                                                  # Ratio of the distance to the vehicle ahead to keep
        self._dt = opt_dict.get('dt', 1.0 / 20.0)                                                                                   # Time between simulation steps
        self._follow_speed_limits = opt_dict.get('follow_speed_limits', False)                                                      # Flag to follow the speed limits
        self._max_brake = opt_dict.get('max_brake', 0.3)                                                                            # Maximum brake applied to the vehicle (0 to 1)
        self._max_steer = opt_dict.get('max_steering', 0.8)                                                                         # Maximum steering applied to the vehicle (-1 to 1)
        self._max_throt = opt_dict.get('max_throttle', 0.75)                                                                        # Maximum throttle applied to the vehicle (0 to 1)
        self._offset = opt_dict.get('offset', 0)                                                                                    # Distance between the route waypoints and the center of the lane
        self._sampling_radius = opt_dict.get('sampling_radius', 2.0)                                                                # Distance between the waypoints part of the plan
        self._target_speed = opt_dict.get('target_speed', 20.0)                                                                     # Desired cruise speed in Km/h

        # Initializing controller
        self._init_controller()

    def _init_controller(self) -> None:
        """
        Controller initialization. Creates the lateral and longitudinal PID controllers.
        """
        self._vehicle_controller = VehicleController(self._vehicle,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

        # Compute the current vehicle waypoint
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    # ====================================================================
    # -- LocalPlanner Control -------------------------------------------
    # ====================================================================
    def run_step(self, debug=False) -> carla.VehicleControl:
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """
        # If the flag is activated, the max speed is set to the speed limit
        if self._follow_speed_limits:
            self._target_speed = self._vehicle.get_speed_limit()

        # Add more waypoints too few in the horizon queue
        if not self._stop_waypoint_creation and len(self._waypoints_queue) < self._min_waypoint_queue_length:
            self.__compute_next_waypoints(k=self._min_waypoint_queue_length)

        # Purge the queue of obsolete waypoints that are too close to the vehicle
        veh_location = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle) / 3.6
        self._min_distance = self._base_min_distance + self._distance_ratio * vehicle_speed

        # Remove waypoints that are too close to the vehicle
        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:
            # Don't remove the last waypoint until very close by
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1        
            # Otherwise, remove waypoints that are too close to the vehicle
            else:
                min_distance = self._min_distance
            # Remove the waypoint if it is too close to the vehicle
            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            # If the waypoint is not close to the vehicle, then stop removing waypoints
            else:
                break

        # If waypoints were removed, update the queue by removing the first 'num_waypoint_removed' waypoints
        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        direction = self.target_road_option if self.target_road_option is not None else RoadOption.LANEFOLLOW
                    
        return control

    # ====================================================================
    # -- LocalPlanner Public Methods ------------------------------------
    # ====================================================================
    def draw_waypoints(self, z = 0.2, size = 0.1, color = carla.Color(0, 255, 0)):
        """
        Draw a list of waypoints at a certain height given in z.

            :param z: height in meters
            :param size: size of the waypoint
            :param color: color of the waypoint
        """
        waypoints = [self.target_waypoint]
        for wpt in waypoints:
            wpt_t = wpt.transform
            begin = wpt_t.location + carla.Location(z=z)
            angle = math.radians(wpt_t.rotation.yaw)
            end = begin + carla.Location(x = math.cos(angle), y = math.sin(angle))
            self._vehicle.get_world().debug.draw_point(end, size = size, life_time = 3.0, color = color)
    
    def done(self) -> bool:
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0

    def follow_speed_limits(self, value : bool = True) -> None:
        """
        Activates a flag that makes the max speed dynamically vary according to the spped limits

        :param value (bool): flag to activate the dynamic speed limits 
        :return:
        """
        self._follow_speed_limits = value

    def get_incoming_waypoint_and_direction(self, steps : int = 3) -> tuple:
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        # If the queue is empty, return None
        if len(self._waypoints_queue) > steps:
            return self._waypoints_queue[steps]
        # If the queue is not empty, return the last waypoint
        else:
            try:
                wpt, direction = self._waypoints_queue[-1]
                return wpt, direction
            except IndexError as i:
                return None, RoadOption.VOID    

    def get_plan(self) -> deque:
        """Returns the current plan of the local planner"""
        return self._waypoints_queue

    def reset_vehicle(self) -> None:
        """Reset the vehicle to its initial conditions"""
        self._vehicle = None

    def set_global_plan(self, current_plan : any, stop_waypoint_creation : bool = True, clean_queue : bool = True) -> None:
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs
        The 'clean_queue` parameter erases the previous plan if True, otherwise, it adds it to the old one
        The 'stop_waypoint_creation' flag stops the automatic creation of random waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :param stop_waypoint_creation: bool
        :param clean_queue: bool
        :return:
        """
        # Clean the queue if the flag is activated
        if clean_queue:
            self._waypoints_queue.clear()
        
        # Remake the waypoints queue if the new plan has a higher length than the queue
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue
            
        # Add the new plan to the queue
        for elem in current_plan:
            self._waypoints_queue.append(elem)

        # Set the flag to stop the automatic creation of waypoints
        self._stop_waypoint_creation = stop_waypoint_creation
        self._vehicle_controller.setWaypoints(self._waypoints_queue)
            
    def set_lateral_offset(self, offset : float) -> None:
        '''
        Set the lateral offset of the controller.
        '''
        self._vehicle_controller._lat_controller.offset = offset
        
    def set_overtake_plan(self, overtake_plan : list, overtake_distance : float) -> list:
        """ 
        Adds an overtake plan to the local planner. The overtake plan is a list of [carla.Waypoint, RoadOption] pairs.
        The overtake distance is the distance to the end of the overtake plan at which the vehicle should return to the 
        normal plan. So, the overtake plan is added to the local planner until this distance is reached.
        
        NOTE: The queue contains a list of [carla.Waypoint, RoadOption] pairs. The RoadOption is used to determine the
        type of connection between the current waypoint and the next waypoint. So, we are interested only at the first 
        element of the pair.
        
        :param overtake_plan: list of (carla.Waypoint, RoadOption)
        :param overtake_distance: float
        
        :return: list of (carla.Waypoint, RoadOption)
        """
        def get_waypoint(distance: float) -> carla.Waypoint:
            """
            Get the waypoint at a distance ahead
            """
            try:
                wpt = list(self._waypoints_queue[0][0].next(distance))[0]
                return wpt
            except IndexError as i:
                return None        
          
        def get_waypoint_index(wp2: carla.Waypoint) -> int:
            """
            This function iterates through the waypoints in the plan and breaks the loop once a waypoint
            outside the sampling radius is found after finding a waypoint within the radius. In other words, 
            it returns the index of the next waypoint after the last waypoint within the sampling radius.
            
                :param wp2 (carla.Waypoint): waypoint to search in the plan
                :return (int): index of the waypoint in the plan
            """
            # Get the length of the waypoints queue
            waypoint_length = len(self._waypoints_queue)

            idx = 0
            # The first while loop iterates through the waypoints in the plan until it finds a waypoint whose distance from 
            # the given waypoint is greater than the sampling radius.
            while (idx < waypoint_length and self._waypoints_queue[idx][0].transform.location.distance(
                    wp2.transform.location) > self._sampling_radius):
                idx += 1
            # The second while loop continues from the point where the first loop stopped and iterates through the waypoints until it finds
            # a waypoint whose distance from the given waypoint is less than the sampling radius.
            while (idx < waypoint_length and self._waypoints_queue[idx][0].transform.location.distance(
                    wp2.transform.location) < self._sampling_radius):
                idx += 1
                
            return idx

        # Get the first waypoint at the overtake distance
        end_overtake_wp = get_waypoint(overtake_distance)
        # If the overtake plan is empty, return the current plan
        if not end_overtake_wp:
            return list(self._waypoints_queue)
                
        # Get the index of the end overtake waypoint in the plan
        idx = get_waypoint_index(end_overtake_wp)
         
        # Return the overtake plan
        overtake_plan.extend(list(self._waypoints_queue)[idx:])
        return overtake_plan
                                
    def set_speed(self, speed: float) -> None:
        """
        Changes the target speed

            :param speed: new target speed in Km/h
            :return:
        """
        if self._follow_speed_limits:
            print("WARNING: The max speed is currently set to follow the speed limits. "
                  "Use 'follow_speed_limits' to deactivate this")
        self._target_speed = speed

    # ====================================================================
    # -- LocalPlanner Private Methods -----------------------------------
    # ====================================================================
    def __compute_next_waypoints(self, k : int = 1) -> None:
        """
        Add new waypoints to the trajectory queue. The number of waypoints to be added is defined by the user.

            :param k: how many waypoints to compute
        """
        # Check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        # Compute the next waypoints to be added to the queue
        for _ in range(k):
            # Get the last waypoint in the queue
            last_waypoint = self._waypoints_queue[-1][0]
            # Get the next waypoints and the road options
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            # If there are no waypoints, break the loop
            if len(next_waypoints) == 0:
                break
            # If there is only one waypoint, then follow the lane
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            # If there are multiple waypoints, then choose a random option
            else:
                # Random choice between the possible options
                road_options_list = self.retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]
            # Append the new waypoint to the queue
            self._waypoints_queue.append((next_waypoint, road_option))

    # ====================================================================
    # -- LocalPlanner Static Methods ------------------------------------
    # ====================================================================
    @staticmethod
    def compute_connection(current_waypoint : carla.Waypoint, next_waypoint : carla.Waypoint, threshold : float = 35.0) -> RoadOption:
        """
        Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
        (next_waypoint).

        :param current_waypoint: active waypoint
        :param next_waypoint: target waypoint
        :return: the type of topological connection encoded as a RoadOption enum:
                RoadOption.STRAIGHT
                RoadOption.LEFT
                RoadOption.RIGHT
        """
        # Get the yaw angle of the next waypoint
        n = next_waypoint.transform.rotation.yaw        
        n = n % 360.0                                       
        # Get the yaw angle of the current waypoint
        c = current_waypoint.transform.rotation.yaw
        c = c % 360.0

        # Compute the difference in angle between the current and next waypoint
        diff_angle = (n - c) % 180.0
        if diff_angle < threshold or diff_angle > (180 - threshold):
            return RoadOption.STRAIGHT
        elif diff_angle > 90.0:
            return RoadOption.LEFT
        else:
            return RoadOption.RIGHT
        
    @staticmethod   
    def retrieve_options(list_waypoints : list, current_waypoint : carla.Waypoint) -> list:
        """
        Compute the type of connection between the current active waypoint and the multiple waypoints present in
        list_waypoints. The result is encoded as a list of RoadOption enums.

        :param list_waypoints: list with the possible target waypoints in case of multiple options
        :param current_waypoint: current active waypoint
        :return: list of RoadOption enums representing the type of connection from the active waypoint to each
                candidate in list_waypoints
        """
        options = []
        for next_waypoint in list_waypoints:
            # This is needed because something we are linking to the beggining of an intersection, 
            # therefore the variation in angle is small. So, we need to check the next waypoint.
            next_next_waypoint = next_waypoint.next(3.0)[0]
            link = LocalPlanner.compute_connection(current_waypoint, next_next_waypoint)
            options.append(link)

        return options