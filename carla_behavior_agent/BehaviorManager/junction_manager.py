# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

import carla
import math
import numpy as np

from enum import IntEnum
from typing import List

from basic_agent import BasicAgent
from misc import *

from local_planner import RoadOption

# ====================================================================
# -- RoadOption Enum -------------------------------------------------
# ====================================================================
class JunctionOption(IntEnum):
    """
    JunctionOption class that provides the following scenarios:
    
    - VOID: the vehicle cannot turn left, right or go front
    - FRONTLEFT: the vehicle is going front and can turn left
    - FRONTRIGHT: the vehicle is going front and can turn right
    - LEFTRIGHT: the vehicle can turn left and right
    - ALL: the vehicle can turn left, right and go front
    
    These scenarios are used to categorize the intersection scenario.
    """
    VOID = -1
    FRONTLEFT = 0
    FRONTRIGHT = 1
    LEFTRIGHT = 2
    FRONTLEFTRIGHT = 3
    
# ====================================================================
# -- JunctionManager -------------------------------------------------
# ====================================================================
class JunctionManager(BasicAgent):
    """
    JunctionManager class that provides the logic to perform the junction maneuver. In particular, the class 
    provides the logic to perform the following actions:
    
    - Check if the junction is free from vehicles.
    - Categorize the junction scenario.
    - Perform the junction maneuver.
    - Perform the emergency stop.
    - Perform the junction maneuver.
    
    The class provides the following methods:
        - run_step: Function that collects the logic and behavior to be adopted in the intersection.
        - in_junction: Check if the vehicle is in the junction.
        - in_junction_cnt: Get the counter of the junction.
        - _stop_cnt: Get the counter of the stop.
        - _stop_cnt: Set the counter of the stop.
    
    The class provides the following private methods:
        - __cross_junction: This function is called when the vehicle is ready to cross the junction. It returns the control to be applied at the junction.
        - __junction_is_free: This function checks if the junction is free from vehicles. It returns True if the junction is free from vehicles, False otherwise.
        - __collect_entry_points: This function takes as input a waypoint that is part of an intersection and returns the entry points of the intersection.
        - __directional_neighbors: This function categorizes the neighbors of a given point according to the direction relative to the reference point. The function returns a python dictionary in which neighbors are categorized according to the direction relative to the reference point.
        - __junction_entries_oriented: Wrapper for "directional neighbors".
        - __junction_categorize: This function categorizes the intersection scenario given the reference vehicle and the junction at which is the vehicle. In particular, it returns the following scenarios: FRONTLEFT, FRONTRIGHT, LEFTRIGHT, ALL, VOID.
    
    The class provides the following static methods:
        - is_vehicle_turning_right: Check if the vehicle is turning right.
        - is_vehicle_turning_left: Check if the vehicle is turning left.
        - is_vehicle_going_straight: Check if the vehicle is going straight.
        
    """

    # ====================================================================
    # -- JunctionManager Init --------------------------------------------
    # ====================================================================
    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)

        self._in_junction_cnt = 0
        self._in_junction = False
        self._junction_point = None  
        self._junction_type = -1
        self._stop_cnt = 0
        self._stuck_time_in_junction = 2                                        # seconds
    
    def run_step(self, local_planner, look_ahead_steps : int) -> bool:
        """Function that collects the logic and behavior to be adopted in the intersection

            :param local_planner (LocalPlanner): local planner to be used to get the plan.
            
            :return bool: True if the vehicle can cross the junction, False otherwise.
        """        
        # Update the local planner
        self._local_planner = local_planner
        # Get the direction of the vehicle
        _, direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps = look_ahead_steps
        )
        # Comes into the intersection and stops for _stuck_time_in_junction seconds
        if self._stop_cnt < (self._stuck_time_in_junction / self._world.get_snapshot().timestamp.delta_seconds):
            print('--- [JUNCTION] waiting cross')
            self._stop_cnt += 1
            return False

        print('--- [JUNCTION] evaluate intersection: {}'.format(self._in_junction))
        
        # If not in junction, find the junction point
        if not self._in_junction:
            print("--- [JUNCTION] entering junction")
            # Get the waypoint of the ego vehicle
            ego_wp = self._map.get_waypoint(self._vehicle.get_location())

            # Find the index of the next junction
            idx = 1
            while not ego_wp.next(idx)[0].is_junction:
                idx += 1
        
            # Save the junction point found for future reference
            junction_wp = ego_wp.next(idx)[0]
            self._junction_point = junction_wp
        else:
            # Retrieve the junction point
            junction_wp = self._junction_point

        # Get the junction from the waypoint
        junction = junction_wp.get_junction()

        # If the junction is not found, stop the vehicle
        if not junction:
            print("--- [JUNCTION] junction not found")
            return True
        
        # Get the pivot point of the junction. This corresponds to the center of the junction.
        pivot = carla.Transform(junction.bounding_box.location, carla.Rotation())
        # Get the radius of the junction. This corresponds to the distance between the center of the junction and the corner of the junction.
        junction_radius = math.sqrt(junction.bounding_box.extent.y ** 2 + junction.bounding_box.extent.x ** 2)
        # Get the vehicles in the junction
        vehicles_in_junction = self._get_ordered_vehicles(pivot, junction_radius)
        vehicles_in_junction = [
            vehicle for vehicle in vehicles_in_junction
            if self._map.get_waypoint(vehicle.get_location()).junction_id == junction.id
        ]
        
        print("[JUNCTION] Vehicles in junction: ", vehicles_in_junction)
        
        # # check whether the junction is free, inside 
        # if not self.__junction_is_free(junction):
        #     print("--- [JUNCTION] junction is not free")
        #     return False
        
        # Get the junction type (front_right, left_right, front_left, all)
        self._junction_type = self.__junction_categorize(junction_wp, junction)
        print("--- [JUNCTION] type: {} - Outgoing direction: {}".format(JunctionOption(self._junction_type).name, RoadOption(direction).name))

        # Get the entry points of the junction
        entry_wps = self.__collect_entry_points(junction)

        # Get the outgoing direction of the vehicle at the junction. This is used to determine the behavior at the junction.       
        pivot = carla.Transform(junction.bounding_box.location, junction_wp.transform.rotation)
        oriented_entries = self.__directional_neighbors(pivot, entry_wps)

        # SCENARIO 0: VOID JUNCTION
        if self._junction_type == JunctionOption.VOID:
            return True

        # SCENARIO 1: FRONT LEFT JUNCTION
        elif self._junction_type == JunctionOption.FRONTLEFT:
            print("--- [JUNCTION] FRONT LEFT JUNCTION")
            # Get the front waypoint
            wp_front = oriented_entries['front'][0]
            print("wp_front: ", wp_front)
            # Get the left waypoint
            wp_left = oriented_entries['left'][0]
            print("wp_left: ", wp_left)
            # Get the vehicles in front and left
            vehicle_front = self._get_ordered_vehicles(wp_front, 9)
            vehicle_left = self._get_ordered_vehicles(wp_left, 9)
            print("vehicle_front: ", vehicle_front)
            print("vehicle_left: ", vehicle_left)
            # Get the first vehicle in front and left
            vehicle_front = vehicle_front[0] if vehicle_front else None
            vehicle_left = vehicle_left[0] if vehicle_left else None
            
            print("ALL JUNCTIONS CONDITION")
            print("JunctionManager.is_vehicle_going_straight(vehicle_front): ", JunctionManager.is_vehicle_going_straight(vehicle_front))
            print("JunctionManager.is_vehicle_going_straight(vehicle_left): ", JunctionManager.is_vehicle_going_straight(vehicle_left))
            print("")
            print("JunctionManager.is_vehicle_turning_left(vehicle_front): ", JunctionManager.is_vehicle_turning_left(vehicle_front))
            print("JunctionManager.is_vehicle_turning_left(vehicle_left): ", JunctionManager.is_vehicle_turning_left(vehicle_left))
            print("")
            print("JunctionManager.is_vehicle_turning_right(vehicle_front): ", JunctionManager.is_vehicle_turning_right(vehicle_front))
            print("JunctionManager.is_vehicle_turning_right(vehicle_left): ", JunctionManager.is_vehicle_turning_right(vehicle_left))
            
            # If there are no vehicles in front and left, cross the junction
            if not vehicle_front and not vehicle_left:
                print("--- [JUNCTION] no vehicles in front and left")
                return True
            
            # If the ego vehicle is going left we need to check the vehicle in front and on the left.
            elif direction == RoadOption.LEFT:
                if not JunctionManager.is_vehicle_going_straight(vehicle_front) and not JunctionManager.is_vehicle_turning_right(vehicle_front) and  not JunctionManager.is_vehicle_turning_left(vehicle_left):
                    print("--- [JUNCTION] Ego vehicle is going left and no vehicles can block the junction")
                    return True
            # If the ego vehicle is going front we need to check the vehicle on the left. We are not interested in the vehicle in front.
            elif direction == RoadOption.STRAIGHT:
                # If the vehicle on the left is not turning left, cross the junction
                if not JunctionManager.is_vehicle_turning_left(vehicle_left):
                    print("--- [JUNCTION] Ego vehicle is going front and no vehicles can block the junction")
                    return True
            print("--- [JUNCTION] SCENARIO 3: EMERGENCY STOP")
            return False 

        # SCENARIO 2: FRONT RIGHT JUNCTION
        elif self._junction_type == JunctionOption.FRONTRIGHT:
            print("--- [JUNCTION] FRONT RIGHT JUNCTION")
            # Get the front waypoint
            wp_front = oriented_entries['front'][0]
            # Get the right waypoint
            wp_right = oriented_entries['right'][0]
            # Get the vehicles in front and right
            vehicle_front = self._get_ordered_vehicles(wp_front, 9)
            vehicle_right = self._get_ordered_vehicles(wp_right, 9)
            print("vehicle_front: ", vehicle_front)
            print("vehicle_right: ", vehicle_right)
            # Get the first vehicle in front and right
            vehicle_front = vehicle_front[0] if vehicle_front else None
            vehicle_right = vehicle_right[0] if vehicle_right else None

            print("ALL JUNCTIONS CONDITION")
            print("JunctionManager.is_vehicle_going_straight(vehicle_front): ", JunctionManager.is_vehicle_going_straight(vehicle_front))
            print("JunctionManager.is_vehicle_going_straight(vehicle_right): ", JunctionManager.is_vehicle_going_straight(vehicle_right))
            print("")
            print("JunctionManager.is_vehicle_turning_left(vehicle_front): ", JunctionManager.is_vehicle_turning_left(vehicle_front))
            print("JunctionManager.is_vehicle_turning_left(vehicle_right): ", JunctionManager.is_vehicle_turning_left(vehicle_right))
            print("")
            print("JunctionManager.is_vehicle_turning_right(vehicle_front): ", JunctionManager.is_vehicle_turning_right(vehicle_front))
            print("JunctionManager.is_vehicle_turning_right(vehicle_right): ", JunctionManager.is_vehicle_turning_right(vehicle_right))

            # If there are no vehicles in front and right, cross the junction
            if not vehicle_front and not vehicle_right:
                print("--- [JUNCTION] no vehicles in front and right")
                return True
                                    
            # If the ego vehicle is going right we need to check the vehicle in front. We are not interested in the vehicle on the right. 
            elif direction == RoadOption.RIGHT:
                # If the vehicle in front is not turning right, cross the junction
                if not JunctionManager.is_vehicle_turning_left(vehicle_front) or (JunctionManager.is_vehicle_turning_left(vehicle_right) and get_speed(vehicle_right) > 0.1):
                    print("--- [JUNCTION] Ego vehicle is going right and no vehicles can block the junction")
                    return True
            # If the ego vehicle is going front we need to check the vehicle on the right and in front.
            elif direction == RoadOption.STRAIGHT:
                # If the vehicle on the right is not turning right, cross the junction
                if not JunctionManager.is_vehicle_turning_left(vehicle_right) and not JunctionManager.is_vehicle_turning_right(vehicle_right) and not JunctionManager.is_vehicle_turning_left(vehicle_front):
                    print("--- [JUNCTION] Ego vehicle is going front and no vehicles can block the junction")
                    return True
            print("--- [JUNCTION] SCENARIO 1: EMERGENCY STOP")
            return False
        
        # SCENARIO 3: LEFT RIGHT JUNCTION
        elif self._junction_type == JunctionOption.LEFTRIGHT:
            print("--- [JUNCTION] LEFT RIGHT JUNCTION")
            # Get the left waypoint
            wp_left = oriented_entries['left'][0]
            # Get the right waypoint
            wp_right = oriented_entries['right'][0]
            # Get the vehicles in left and right
            vehicle_left = self._get_ordered_vehicles(wp_left, 9)
            vehicle_right = self._get_ordered_vehicles(wp_right, 9)
            print("vehicle_left: ", vehicle_left)
            print("vehicle_right: ", vehicle_right)
            
            # Get the first vehicle in left and right
            vehicle_left = vehicle_left[0] if vehicle_left else None
            vehicle_right = vehicle_right[0] if vehicle_right else None
            
            print("ALL JUNCTIONS CONDITION")
            print("JunctionManager.is_vehicle_going_straight(vehicle_left): ", JunctionManager.is_vehicle_going_straight(vehicle_left))
            print("JunctionManager.is_vehicle_going_straight(vehicle_right): ", JunctionManager.is_vehicle_going_straight(vehicle_right))
            print("")
            print("JunctionManager.is_vehicle_turning_left(vehicle_left): ", JunctionManager.is_vehicle_turning_left(vehicle_left))
            print("JunctionManager.is_vehicle_turning_left(vehicle_right): ", JunctionManager.is_vehicle_turning_left(vehicle_right))
            print("")
            print("JunctionManager.is_vehicle_turning_right(vehicle_left): ", JunctionManager.is_vehicle_turning_right(vehicle_left))
            print("JunctionManager.is_vehicle_turning_right(vehicle_right): ", JunctionManager.is_vehicle_turning_right(vehicle_right))
            
            # If there are no vehicles in left and right, cross the junction
            if not vehicle_left and not vehicle_right:
                print("--- [JUNCTION] no vehicles in left and right")
                return True
            
            # If the ego vehicle is going left we need to check the vehicle on the right and on the left.
            if direction == RoadOption.LEFT:
                if not JunctionManager.is_vehicle_going_straight(vehicle_left) and not JunctionManager.is_vehicle_going_straight(vehicle_right) and not JunctionManager.is_vehicle_turning_left(vehicle_right):
                    print("--- [JUNCTION] Ego vehicle is going on the left and no vehicles can block the junction. We can cross the junction.")
                    return True
                elif JunctionManager.is_vehicle_turning_right(vehicle_left) and vehicle_right in vehicles_in_junction:
                    print("--- [JUNCTION] Ego vehicle is going on the left and the vehicle on the left is turning right while the vehicle on the right is in the junction. We can cross the junction.")
                    return True
            elif direction == RoadOption.RIGHT:
                if not JunctionManager.is_vehicle_going_straight(vehicle_left):
                    print("--- [JUNCTION] Ego vehicle is going on the right and no vehicles can block the junction. We can cross the junction.")
                    return True
                
            return False          
        
        # SCENARIO 4: ALL JUNCTION (FRONT, LEFT, RIGHT)       
        elif self._junction_type == JunctionOption.FRONTLEFTRIGHT:
            print("--- [JUNCTION] ALL JUNCTION")
            # Get the front waypoint
            wp_front = oriented_entries['front'][0]
            print("wp_front: ", wp_front)
            # Get the left waypoint
            wp_left = oriented_entries['left'][0]
            print("wp_left: ", wp_left)
            # Get the right waypoint
            wp_right = oriented_entries['right'][0]
            print("wp_right: ", wp_right)            
            
            # Get the vehicles in front, left and right
            vehicle_front = self._get_ordered_vehicles(wp_front, 9)
            vehicle_left = self._get_ordered_vehicles(wp_left, 9)
            vehicle_right = self._get_ordered_vehicles(wp_right, 9)
            print("vehicle_front: ", vehicle_front)
            print("vehicle_left: ", vehicle_left)
            print("vehicle_right: ", vehicle_right)
            # Get the first vehicle in front, left and right
            vehicle_front = vehicle_front[0] if vehicle_front else None
            vehicle_left = vehicle_left[0] if vehicle_left else None
            vehicle_right = vehicle_right[0] if vehicle_right else None
            
            print("ALL JUNCTIONS CONDITION")
            print("JunctionManager.is_vehicle_going_straight(vehicle_front): ", JunctionManager.is_vehicle_going_straight(vehicle_front))
            print("JunctionManager.is_vehicle_going_straight(vehicle_left): ", JunctionManager.is_vehicle_going_straight(vehicle_left))
            print("JunctionManager.is_vehicle_going_straight(vehicle_right): ", JunctionManager.is_vehicle_going_straight(vehicle_right))
            print("")
            print("JunctionManager.is_vehicle_turning_left(vehicle_front): ", JunctionManager.is_vehicle_turning_left(vehicle_front))
            print("JunctionManager.is_vehicle_turning_left(vehicle_left): ", JunctionManager.is_vehicle_turning_left(vehicle_left))
            print("JunctionManager.is_vehicle_turning_left(vehicle_right): ", JunctionManager.is_vehicle_turning_left(vehicle_right))
            print("")
            print("JunctionManager.is_vehicle_turning_right(vehicle_front): ", JunctionManager.is_vehicle_turning_right(vehicle_front))
            print("JunctionManager.is_vehicle_turning_right(vehicle_left): ", JunctionManager.is_vehicle_turning_right(vehicle_left))
            print("JunctionManager.is_vehicle_turning_right(vehicle_right): ", JunctionManager.is_vehicle_turning_right(vehicle_right))
            
            # If there are no vehicles in front, left and right, cross the junction
            if not vehicle_front and not vehicle_left and not vehicle_right:
                print("--- [JUNCTION] no vehicles in front, left and right")
                return True
            
            # If the ego vehicle is going front we need to check the vehicle in front, on the left and on the right.
            if direction == RoadOption.STRAIGHT:
                if not JunctionManager.is_vehicle_turning_left(vehicle_left) and not JunctionManager.is_vehicle_turning_right(vehicle_right) and not JunctionManager.is_vehicle_turning_left(vehicle_front) \
                    and not JunctionManager.is_vehicle_going_straight(vehicle_left) and not JunctionManager.is_vehicle_going_straight(vehicle_right):                
                    print("--- [JUNCTION] Ego vehicle is going front and no vehicles can block the junction. We can cross the junction.")
                    return True
                elif JunctionManager.is_vehicle_turning_right(vehicle_left) and vehicle_left in vehicles_in_junction:
                    print("--- [JUNCTION] Ego vehicle is going front and the vehicle on the left is turning right. We can cross the junction.")
                    return True
                elif (JunctionManager.is_vehicle_going_straight(vehicle_front) or JunctionManager.is_vehicle_turning_right(vehicle_front)) and vehicle_front in vehicles_in_junction:
                    print("--- [JUNCTION] Ego vehicle is going front and the vehicle in front is going straight or turning right. We can cross the junction.")
                    return True
            # If the ego vehicle is going left we need to check the vehicle in front, on the left and on the right.
            elif direction == RoadOption.LEFT:
                if not JunctionManager.is_vehicle_going_straight(vehicle_front) and not JunctionManager.is_vehicle_going_straight(vehicle_left) and not JunctionManager.is_vehicle_going_straight(vehicle_right) \
                    and not JunctionManager.is_vehicle_turning_right(vehicle_front) and not JunctionManager.is_vehicle_turning_left(vehicle_left) and not JunctionManager.is_vehicle_turning_left(vehicle_right): 
                    print("--- [JUNCTION] Ego vehicle is going left and no vehicles can block the junction")
                    return True
                elif (JunctionManager.is_vehicle_turning_left(vehicle_front) or JunctionManager.is_vehicle_turning_right(vehicle_front) ) and vehicle_front in vehicles_in_junction:
                    print("--- [JUNCTION] Ego vehicle is going left and the vehicle in front is turning left or right. We can cross the junction.")
                    return True
                elif JunctionManager.is_vehicle_turning_right(vehicle_left) and vehicle_left in vehicles_in_junction:
                    print("--- [JUNCTION] Ego vehicle is going left and the vehicle on the left is turning right. We can cross the junction.")
                    return True
                elif JunctionManager.is_vehicle_turning_right(vehicle_right) and vehicle_right in vehicles_in_junction:
                    print("--- [JUNCTION] Ego vehicle is going left and the vehicle on the right is turning right. We can cross the junction.")
                    return True
            # If the ego vehicle is going right we need to check the vehicle in front, on the left and on the right.
            elif direction == RoadOption.RIGHT:
                if not JunctionManager.is_vehicle_going_straight(vehicle_left) and not JunctionManager.is_vehicle_turning_left(vehicle_front):
                    print("--- [JUNCTION] Ego vehicle is going right and no vehicles can block the junction. We can cross the junction.")
                    return True
                
            print("--- [JUNCTION] SCENARIO 4: EMERGENCY STOP")
            return False
        
        print("--- [JUNCTION] EMERGENCY STOP")
        return False

    @property
    def in_junction(self) -> bool:
        """Check if the vehicle is in the junction"""
        return self._in_junction
    
    @in_junction.setter
    def in_junction(self, value: bool) -> None:
        """Set the vehicle in the junction"""
        self._in_junction = value
        
    @property
    def in_junction_cnt(self) -> int:
        """Get the counter of the junction"""
        return self._in_junction_cnt
    
    @in_junction_cnt.setter
    def in_junction_cnt(self, value: int) -> None:
        """Set the counter of the junction"""
        self._in_junction_cnt = value
        
    @property
    def stop_cnt(self) -> int:
        """Get the counter of the stop"""
        return self._stop_cnt
    
    @stop_cnt.setter
    def stop_cnt(self, value: int) -> None:
        """Set the counter of the stop"""
        self._stop_cnt = value
        
    @property
    def junction_point(self) -> carla.Waypoint:
        """Get the junction point"""
        return self._junction_point
    
    @junction_point.setter
    def junction_point(self, value: carla.Waypoint) -> None:
        """Set the junction point"""
        self._junction_point = value

    @property
    def junction_type(self) -> int:
        """Get the junction type"""
        return self._junction_type
    
    @junction_type.setter
    def junction_type(self, value: int) -> None:
        """Set the junction type"""
        self._junction_type = value

    # ====================================================================
    # -- JunctionManager Private Methods ---------------------------------
    # ====================================================================
    def __collect_entry_points(self, junction: carla.Waypoint) -> set:
        """
        This function takes as input a waypoint that is part of an intersection and returns the entry points of the intersection.
        NOTE: The methods junction.get_waypoints(carla.LaneType.Driving) returns a list of tuples, each tuple contains two waypoints, 
        the first is the entry point and the second is the exit point of the intersection. In other words, every tuple on the list contains
        first an initial and then a final waypoint within the intersection boundaries that describe the beginning and the end of said lane 
        along the junction. Lanes follow their OpenDRIVE definitions so there may be many different tuples with the same starting waypoint 
        due to possible deviations, as this are considered different lanes. 
        
            :param junction (carla.Waypoint): waypoint that is part of the intersection.

            :return entry (set): set of entry points of the intersection.
        """
        # Get the waypoints of the junction that are part of the driving lane
        junction_wps = junction.get_waypoints(carla.LaneType.Driving)
        
        # Initialize the entry and exit sets of waypoints.
        # NOTE: We use the set data structure to avoid duplicates. This is due to the fact that the same waypoint can be part of multiple lanes.
        entry_wps = set()
        seen_coords = set()
        for begin, _ in junction_wps:
            # Create a unique key for a waypoint based on its rounded coordinates.
            key = (round(begin.transform.location.x, 2), round(begin.transform.location.y, 2), round(begin.transform.location.z, 2))
            # Check if the key is already in the set of seen coordinates. If it is, skip the waypoint.
            if key in seen_coords:
                continue
            # Add the waypoint to the set of entry points and the set of seen coordinates.
            entry_wps.add(begin)
            seen_coords.add(key)

        return entry_wps
    
    def __directional_neighbors(self, point: carla.Transform, neighbors: List[carla.Waypoint], direction: str = None) -> dict:
        """
        This function categorizes the neighbors of a given point according to the direction relative to the reference point.
        The function returns a python dictionary in which neighbors are categorized according to the direction relative to the reference point.
        
        The directions are defined as follows:
        
        - left: the neighbor is on the left side of the reference point
        - right: the neighbor is on the right side of the reference point
        - front: the neighbor is in front of the reference point
        - back: the neighbor is behind the reference point 
        
            :param point (carla.Transform): reference point (position + orientation).
            :param neighbors (List[carla.Waypoint]): neighbors list of the reference point.
            :param direction (str, optional): if you are interested in neighbors in a specific direction (left, right, front, back) specify the direction to get directly the list of neighbors in the direction of interest. Defaults to None.
            
            :return dir_neighbors (dict): dictionary in which neighbors are categorized according to the direction relative to the reference point.
        """
        # Initialize the dictionary of neighbors
        dir_neighbors = {'left': [], 'right': [], 'front': [], 'back': []}

        # Get the position and orientation of the reference point        
        pivot_pos = np.array([point.location.x, point.location.y, point.location.z])
        pivot_yaw = math.radians(point.rotation.yaw)

        for n in neighbors:
            # Get the position of the neighbor
            neighbor = np.array([n.transform.location.x, n.transform.location.y, n.transform.location.z])
            # Compute the vector between the reference point and the neighbor
            v = neighbor - pivot_pos
            # Compute the unit vector along the orientation of the reference point
            u = np.array([np.cos(pivot_yaw), np.sin(pivot_yaw), 0])
            # Compute the vector product between the unit vector along the orientation of the reference point 
            # and the vector between the reference point and the neighbor.
            cross = np.cross(u, v)[2]
            # Compute the scalar product between the unit vector along the orientation of the reference point
            inner = np.inner(u, v)

            # Categorize the neighbor according to the direction relative to the reference point
            if abs(cross) < abs(inner):
                dir_neighbors['front' if inner > 0 else 'back'].append(n)
            else:
                dir_neighbors['left' if cross < 0 else 'right'].append(n)

        return dir_neighbors if direction is None or direction not in dir_neighbors else dir_neighbors[direction]

    def __junction_categorize(self, vehicle_wp: carla.Waypoint, junction: carla.Junction) -> JunctionOption:
        """
        This function categorizes the intersection scenario given the reference vehicle and the junction at which is the vehicle.
        In particular, it returns the following scenarios:
        
        - FRONTLEFT: the vehicle is going front and can turn left
        - FRONTRIGHT: the vehicle is going front and can turn right
        - LEFTRIGHT: the vehicle can turn left and right
        - ALL: the vehicle can turn left, right and go front
        - VOID: the vehicle cannot turn left, right or go front

            :param vehicle_wp (carla.Waypoint): reference vehicle waypoint at the intersection.
            :param junction (carla.Junction): junction at which the vehicle is.
            
            :return junction_type (JunctionOption): junction scenario.
        """
        # Get the waypoints in the junction that are part of the driving lane
        wps_in_junction = junction.get_waypoints(carla.LaneType.Driving)
        
        # Get the outgoing waypoints of the vehicle at the junction. These are the waypoints that are close to the vehicle.
        # NOTE: The distance threshold is set to 3 meters. In other words, we consider the waypoints close to the vehicle 
        # if they are within 3 meters.
        outgoing_wps = []
        outgoing_wps.extend([wp2 for wp1, wp2 in wps_in_junction if dist(wp1, vehicle_wp) < 3])
        outgoing_wps.extend([wp1 for wp1, wp2 in wps_in_junction if dist(wp2, vehicle_wp) < 3])

        # Get the pivot point of the junction. This corresponds to the center of the junction bounding box with the orientation of the vehicle.
        pivot = carla.Transform(junction.bounding_box.location, vehicle_wp.transform.rotation)
        outgoing_wps_oriented = self.__directional_neighbors(pivot, outgoing_wps)

        # Scenario 1: front_left
        if outgoing_wps_oriented['left'] and not outgoing_wps_oriented['right'] and outgoing_wps_oriented['front']: 
            junction_type = JunctionOption.FRONTLEFT
        # Scenario 2: front_right
        elif not outgoing_wps_oriented['left'] and outgoing_wps_oriented['right'] and outgoing_wps_oriented['front']:
            junction_type = JunctionOption.FRONTRIGHT
        # Scenario 3: left_right
        elif outgoing_wps_oriented['left'] and outgoing_wps_oriented['right'] and not outgoing_wps_oriented['front']:
            junction_type = JunctionOption.LEFTRIGHT
        # Scenario 4: all
        elif outgoing_wps_oriented['left'] and outgoing_wps_oriented['right'] and outgoing_wps_oriented['front']:
            junction_type = JunctionOption.FRONTLEFTRIGHT
        else:
            junction_type = JunctionOption.VOID
            
        return junction_type
       
    # ====================================================================
    # -- JunctionManager Static Methods ----------------------------------
    # ====================================================================   
    @staticmethod
    def is_vehicle_going_straight(vehicle):
        """
        Verifica se un veicolo sta andando dritto basandosi sullo stato dei segnali di svolta e sull'orientamento.
        
        Args:
            vehicle: L'oggetto veicolo da verificare.
            waypoint: L'oggetto waypoint da confrontare.
        
        Returns:
            bool: True se il veicolo sta andando dritto, False altrimenti.
        """
        if not vehicle:
            return False
        
        # Verifica lo stato dei segnali di svolta
        light_state = vehicle.get_light_state()
                
        if bool(light_state & carla.libcarla.VehicleLightState.LeftBlinker) or \
        bool(light_state & carla.libcarla.VehicleLightState.RightBlinker):
            return False
        
        return True
        
    @staticmethod
    def is_vehicle_turning_left(vehicle):
        """
        Verifica se il veicolo sta attivando il segnale di svolta a sinistra.
        
        Args:
            vehicle: L'oggetto veicolo da verificare.
        
        Returns:
            bool: True se il segnale di svolta a sinistra è attivato, False altrimenti.
        """
        if not vehicle:
            return False
        
        light_state = vehicle.get_light_state()
        return bool(light_state & carla.libcarla.VehicleLightState.LeftBlinker)
    
    @staticmethod
    def is_vehicle_turning_right(vehicle):
        """
        Verifica se il veicolo sta attivando il segnale di svolta a destra.
        
        Args:
            vehicle: L'oggetto veicolo da verificare.
        
        Returns:
            bool: True se il segnale di svolta a destra è attivato, False altrimenti.
        """
        if not vehicle:
            return False        
        
        light_state = vehicle.get_light_state()
        return bool(light_state & carla.libcarla.VehicleLightState.RightBlinker)