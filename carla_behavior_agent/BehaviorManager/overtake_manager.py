# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

import carla
import math

from basic_agent import BasicAgent
from misc import *

# ====================================================================
# -- OvertakeManager ---------------------------------------------------
# ====================================================================
class OvertakeManager(BasicAgent):
    """
    OvertakeManager class that provides the logic to perform an overtaking maneuver.
    In particular, the class provides the logic to generate a path to overtake the vehicle in front of the ego vehicle.
    
    The class inherits from the BasicAgent class. The BasicAgent class provides the basic functionality to interact with the CARLA simulator.
    
    The class provides the following methods:
        - run_step: function that collects the logic and behavior to be adopted for the overtaking maneuver.
        - in_overtake: function that returns True if the agent is performing an overtaking maneuver, False otherwise.
        - overtake_cnt: function that returns the counter of the overtaking maneuver.
        - overtake_cnt: function that sets the counter of the overtaking maneuver.
        - overtake_ego_distance: function that returns the distance that the agent needs to travel to overtake the obstacles in front of it.
        
    The class provides the following attributes:
        - _overtake_cnt: integer value that indicates the counter of the overtaking maneuver.
        - _in_overtake: boolean value that indicates the status of the overtaking maneuver.
        - _overtake_ego_distance: float value that indicates the distance that the agent needs to travel to overtake the obstacles in front of it.
        
    The class provides the following static methods:
        - get_overtake_distance: function that calculates the distance that the ego vehicle needs to travel to overtake the vehicle in front of it.
        - get_overtake_time: function that calculates the time it takes for the ego vehicle to overtake the vehicle in front of it.
    """

    # ====================================================================
    # -- OvertakeManager Init --------------------------------------------
    # ====================================================================
    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """
        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)

        self._overtake_cnt = 0
        self._in_overtake = False
        self._overtake_ego_distance = 0
    
    def run_step(
        self, 
        object_to_overtake: carla.Actor, 
        ego_vehicle_wp: carla.Waypoint,
        distance_same_lane: float = 1,
        distance_other_lane: float = 0,
        distance_from_object: float = 18,
        speed_limit: float = 50
    ) -> None:
        """Function that collects the logic and behavior to be adopted for the overtaking maneuver

            :param object_to_overtake: the vehicle to overtake (carla.Actor)
            :param ego_vehicle_wp: the waypoint of the ego vehicle (carla.Waypoint)
            :param distance_on_same_lane: the distance to travel on the same lane before changing lane (float)
            :param distance_on_other_lane: the distance to travel on the other lane before changing lane (float)
            :param distance_from_object: the distance from the object to overtake (float)
            :param speed_limit: the speed limit of the road (float)
        """
        # Get the distance on the other lane if it is not provided. This is the case when the agent is overtaking a parked vehicle.
        if not distance_other_lane:
            distance_other_lane=self.__get_distance_other_lane(object_to_overtake, 30)

        # Get the total lenght of ego vehicle. 
        # NOTE: The extent of the bounding box of the ego vehicle is the half of the lenght of the vehicle.
        vehicle_length = self._vehicle.bounding_box.extent.x 
        lane_width = ego_vehicle_wp.lane_width

        # Get the overtake distance that the agent needs to travel.
        self._overtake_ego_distance, hypotenuse = OvertakeManager.get_overtake_distance(vehicle_length, lane_width, distance_same_lane, distance_other_lane, distance_from_object)

        # Get the overtake time that the agent needs to travel.
        overtake_time = OvertakeManager.get_overtake_time(ego_vehicle=self._vehicle, overtake_distance=self._overtake_ego_distance)
        
        # Check if there is a vehicle in the opposite lane that can obstruct the overtake maneuver.
        # NOTE: We assume that every vehicle in the opposite lane is moving at the speed limit.
        opposite_vehicle_distance = overtake_time * speed_limit / 3.6
        search_distance = self._overtake_ego_distance + opposite_vehicle_distance

        # Check if there is a vehicle in the opposite lane that can obstruct the overtake maneuver.
        opposing_vehicle = self.__opposite_vehicle(ego_wp=ego_vehicle_wp, search_distance=search_distance)

        # 4. Check wheter te ego vehicle will be in an intersection at the end of the overtake
        next_ego_wp = ego_vehicle_wp.next(self._overtake_ego_distance)       
        try:
            # Get the next waypoint of the ego vehicle at 'over_take' distance.
            next_ego_wp = ego_vehicle_wp.next(self._overtake_ego_distance)[0]
            
            # Get the next waypoint of the opposing vehicle at 'search_distance' distance.
            next_opposing_wp = self._map.get_waypoint(opposing_vehicle.get_location()).next(opposite_vehicle_distance)[0]
            
            if next_ego_wp and next_opposing_wp:
                collision = not is_within_distance(target_transform=next_opposing_wp.transform, reference_transform=next_ego_wp.transform, max_distance=30, angle_interval=[0, 90])
        except:
            collision = False
                        
        wp_in_junction = False if not next_ego_wp else next_ego_wp.is_junction
                        
        if not self._overtake_cnt and not (collision or wp_in_junction):
            # Generate the path to overtake the vehicle.
            overtake_path = self._generate_lane_change_path(
                waypoint=ego_vehicle_wp,
                direction='left',
                distance_same_lane=distance_same_lane,
                distance_other_lane=distance_other_lane,
                lane_change_distance=hypotenuse,
                check=False, 
                step_distance=self._sampling_resolution,
            )
            # If the path is not generated, return.
            if not overtake_path:
                return None
                        
            # Update the overtake counter.
            self._overtake_cnt = int(round(overtake_time) / self._world.get_snapshot().timestamp.delta_seconds)
            # Update the overtake status.
            self._in_overtake = True
            
            return overtake_path

    @property
    def in_overtake(self) -> bool:
        """Return True if the agent is performing an overtaking maneuver, False otherwise."""
        return self._in_overtake
    
    @property
    def overtake_cnt(self) -> int:
        """Return the counter of the overtaking maneuver."""
        return self._overtake_cnt
    
    @overtake_cnt.setter
    def overtake_cnt(self, value: int) -> None:
        """Set the counter of the overtaking maneuver."""
        self._overtake_cnt = value
        
    @in_overtake.setter
    def in_overtake(self, value: bool) -> None:
        """Set the status of the overtaking maneuver."""
        self._in_overtake = value

    @property
    def overtake_ego_distance(self) -> float:
        """Return the distance that the agent needs to travel to overtake the obstacles in front of it."""
        return self._overtake_ego_distance

    # ====================================================================
    # -- OvertakeManager Private Methods ---------------------------------
    # ====================================================================
    def __get_distance_other_lane(self, actor: carla.Actor, max_distance: float = 30) -> float:
        """
        This function returns the distance that the agent needs to travel in order to overtake the obstacles (vehicles or static objects) in front of it.
        
            :param vehicle: carla.Actor object that the agent needs to overtake.
            :param max_distance: float value that indicates the maximum distance to search for obstacles in the simulation.
            
            :return distance_other_lane: float value that indicates the distance that the agent needs to travel in order to overtake the obstacles in front of it.
        """
        # Get the length of the vehicle in front of the ego vehicle.
        # NOTE: The 'extent' of the vehicle is the half of the vehicle length.
        actor_length = actor.bounding_box.extent.x

        # Initialize the overtaking distance.
        distance_other_lane = actor_length

        # Get the list of vehicles in the simulation at a distance of 'max_distance' from the ego vehicle.
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_list = [
            v for v in vehicle_list 
            if v.id != actor.id and v.id != self._vehicle.id and dist(v, actor) < max_distance \
            and self._map.get_waypoint(v.get_location()).lane_id == self._map.get_waypoint(actor.get_location()).lane_id
        ]
        
        # Get the list of parked vehicles in the simulation at a distance of 'max_distance' from the ego vehicle.
        vehicle_list = [
            v for v in vehicle_list
            if self._parked_vehicle(v)[1] == True
        ]
        
        previous_vehicle = actor
        print("===GET DISTANCE OTHER LANE===")
        for v in vehicle_list:
            # Get the distance between the current vehicle and the previous vehicle.
            if is_within_distance(target_transform=v.get_transform(), reference_transform=previous_vehicle.get_transform(), max_distance=max_distance, angle_interval=[0, 60]):
                v_distance = compute_distance_from_center(actor1=previous_vehicle, actor2=v, distance=dist(v, previous_vehicle))
            else:
                continue             
            # Update the overtake distance by adding the length of the vehicle in front of the ego vehicle and the distance between the ego vehicle and the vehicle in front of it.
            distance_other_lane += v.bounding_box.extent.x + v_distance
            # Update the previous vehicle.
            previous_vehicle = v
            
            # print("===VEHICLE ID ", v.id)
            # print("===VEHICLE NAME ", v.type_id)
            # print("===VEHICLE DISTANCE ", v_distance)
            # print("===VEHICLE LANES ", self._map.get_waypoint(v.get_location()).lane_id)
            # print("===VEHICLE LENGTH ", v.bounding_box.extent.x)

        return distance_other_lane + 3
          
    def __opposite_vehicle(self, ego_wp: carla.Waypoint = None, search_distance : float = 30) -> carla.Actor:
        '''
        This function returns the vehicle in the opposite lane of the ego vehicle. In particular, it returns the vehicle in the opposite lane 
        of the ego vehicle that is within a distance of 'max_distance' from the ego vehicle.
        
            :param ego_wp (carla.Waypoint): waypoint object of the ego vehicle.
            :param max_distance (float): maximum distance to search for vehicles in the simulation.
            
            :return vehicle (carla.Actor): actor object of the vehicle in the opposite lane of the ego vehicle.
        '''
        def extend_bounding_box(actor: carla.Actor) -> carla.Transform:
            '''
            This function computes the extremum point of the actor in the specified position.
            
                :param actor (carla.Actor): actor object.
                
                :return transform (carla.Transform): transform object of the extremum point of the actor.
            '''
            # Get the waypoint of the actor
            wp = self._map.get_waypoint(actor.get_location())
            transform = wp.transform
            # Get the forward vector of the actor
            forward_vector = transform.get_forward_vector()
            # Extend the bounding box of the actor in the forward direction of a quantity equal to half of the vehicle length
            extent = actor.bounding_box.extent.x
            location = carla.Location(x=extent * forward_vector.x, y=extent * forward_vector.y, )
            transform.location += location
            return transform
        
        # Get all vehicles near ego, ordered and with radius search_distance
        vehicle_list = self._get_ordered_vehicles(self._vehicle, search_distance)
        # Filter out vehicles in the same lane as the ego vehicle. 
        vehicle_list = [
            v for v in vehicle_list 
            if self._map.get_waypoint(v.get_location()).lane_id == ego_wp.lane_id * -1
        ]

        if not vehicle_list:
            return None

        # Extend the bounding box of the ego vehicle in the forward direction of a quantity equal to half of the vehicle length
        ego_front_transform = extend_bounding_box(self._vehicle)

        for vehicle in vehicle_list:
            # Extend the bounding box of the vehicle in the forward direction of a quantity equal to half of the vehicle length
            target_front_transform = extend_bounding_box(vehicle)
            # Check if the vehicle is in the opposite lane of the ego vehicle
            if is_within_distance(target_front_transform, ego_front_transform, search_distance, angle_interval=[0, 90]):
                return vehicle
    
        return None
    
    # ====================================================================
    # -- OvertakeManager Static Methods ------------------------------------
    # ====================================================================    
    @staticmethod
    def get_overtake_distance(
        vehicle_length : float, 
        lane_width : float, 
        distance_same_lane : float, 
        distance_other_lane : float,
        distance_from_obstacle : float
    ) -> float:
        """ 
        This function calculates the distance that the ego vehicle needs to travel to overtake the vehicle in front of it.
        In particular, the distance is calculated using the Pythagorean theorem to approximate the distance that the ego vehicle needs 
        to travel to change lanes.
        
            :param ego_vehicle: the ego vehicle (carla.Vehicle)
            :param lane_width: the width of the lane (float)
            :param distance_on_same_lane: the distance that the ego vehicle needs to travel on the same lane (float)
            :param distance_on_other_lane: the distance that the ego vehicle needs to travel on the other lane (float)
            :param distance_from_obstacle: the distance from the object in front of the ego vehicle (float)
            
            :return: the distance that the ego vehicle needs to travel to overtake the vehicle in front of it and 
            the hypotenuse of the triangle formed by the lenght of the ego vehicle and the lane width (tuple)
        """    
        # Calculate the hypotenuse of the triangle formed by the lenght of the ego vehicle and the lane width
        # NOTE: The hypotenuse approximates the distance that the ego vehicle needs to travel to change lanes.
        hypotenuse = math.sqrt(vehicle_length**2 + lane_width**2)
        
        # Calculate the total distance that the ego vehicle needs to travel to overtake the vehicle in front of it.
        overtake_distance = distance_from_obstacle + distance_same_lane + hypotenuse + distance_other_lane + hypotenuse
        
        return overtake_distance, hypotenuse

    @staticmethod
    def get_overtake_time(
        ego_vehicle : carla.Vehicle,
        overtake_distance : float,
        ) -> float:
        '''
        This function calculates the time it takes for the ego vehicle to overtake the vehicle in front of it.
        In particular, the time is calculated using the uniformly accelerated rectilinear motion equation:
        ====> s = v0*t + 0.5*a*t^2
        from which we can derive the time it takes for the ego vehicle to overtake the vehicle in front of it:
        ====> t = (-v0 + sqrt(v0^2 + 2*a*s)) / a
        
            :param ego_vehicle: the ego vehicle (carla.Vehicle)
            :param overtake_distance: the distance that the ego vehicle needs to travel to overtake the vehicle in front of it (float)
            
            :return: the time it takes for the ego vehicle to overtake the vehicle in front of it (float)
        '''
        # Get the current speed of the ego vehicle
        v0 = get_speed(ego_vehicle) / 3.6                                                                                  # m/s
    
        # Get the acceleration of the ego vehicle
        a = 3.5                                                                                                            # m/s^2            
        
        # Calculate the time it takes for the ego vehicle to overtake the vehicle in front of it.
        overtake_time = (-v0 + math.sqrt(v0 ** 2 + 2 * a * overtake_distance)) / a                                         # s

        return overtake_time