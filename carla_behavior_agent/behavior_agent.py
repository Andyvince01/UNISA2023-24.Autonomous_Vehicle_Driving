# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from BehaviorManager.overtake_manager import OvertakeManager
from BehaviorManager.junction_manager import JunctionManager, JunctionOption

from misc import *
# from utils import configure_logger

# logger = configure_logger()

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='cautious', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)

        # Vehicle information
        self._approach_speed = 10.0                                 # Approach speed of the agent
        self._behavior = \
            Aggressive() if behavior == 'aggressive' else \
            Normal() if behavior == 'normal' else \
            Cautious()            
        self._direction = None                                      # Current direction of the agent
        self._incoming_direction = None                             # Incoming direction of the agent
        self._incoming_waypoint = None                              # Incoming waypoint of the agent
        self._look_ahead_steps = 0                                  # Number of steps to look ahead
        self._min_speed = 5                                         # Minimum speed of the agent
        self._speed = 0                                             # Current speed of the agent
        self._speed_limit = 0                                       # Speed limit of the agent
        self._stuck = False                                         # Flag indicating if the agent is stuck

        # Overtake Manager Instance
        self._overtake_manager = OvertakeManager(self._vehicle, opt_dict)

        # Junction Manager Istance
        self._junction_manager = JunctionManager(self._vehicle, opt_dict)
        self._junction_right_cnt = 0
        
        # Parameters for agent behavior
        self._behavior = Cautious() if behavior == 'cautious' else Aggressive() if behavior == 'aggressive' else Normal()
        self._is_raining = False
                    
    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        # Update the information of the agent.
        self.__update_information(debug=debug)
                                                                
        # Initialize the control of the agent.
        control = None
        
        # Get the location of the agent and the relative waypoint.
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        
        # SCENARIO 1: Red lights and stops behavior
        affected_by_traffic_light = self.traffic_light_manager(vehicle=self._vehicle)
        
        if affected_by_traffic_light:
            print("SCENARIO 1: Red lights and stops behavior!")
            return self.__emergency_stop()
    
        # SCENARIO 2: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)
        if walker_state:
            # Distance is computed from the center of the two cars.
            w_distance = compute_distance_from_center(actor1=self._vehicle, actor2=walker, distance=w_distance)
            print("[SCENARIO 2] Pedestrian {0} is {1} meters away from the ego vehicle!".format(walker, w_distance))

            # Emergency brake if the car is very close.
            if w_distance < self._behavior.min_proximity_threshold:
                print("--- Emergency stop in front of the pedestrian!")
                return self.__emergency_stop()
            else:
                print("--- Reducing speed in front of the pedestrian!")
                self.set_target_speed(self._approach_speed)
                return self._local_planner.run_step()
                    
        # SCENARIO 3: Obstacle avoidance behaviors
        tw_state, tw, tw_distance = self.static_obstacle_manager(waypoint=ego_vehicle_wp, static_element="*static.prop.trafficwarning*")
        cone_state, cone, _ = self.static_obstacle_manager(waypoint=ego_vehicle_wp, static_element="*static.prop.constructioncone*")
        
        # If there is a traffic warning sign, the agent will overtake the obstacle if it is safe to do so. 
        # We give priority to the traffic warning signs over the construction cones.
        if tw_state is True:
            # Distance is computed from the center of the obstacle and the ego vehicle. 
            tw_distance = compute_distance_from_center(actor1=self._vehicle, actor2=tw, distance=tw_distance)
            
            # Overtake the obstacle if it is safe to do so.
            overtake_path = self._overtake_manager.run_step(
                object_to_overtake=tw, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_other_lane=20, distance_from_object=tw_distance, speed_limit = self._speed_limit
            )

            if overtake_path:
                print("[SCENARIO 3] - Overtaking the traffic warning sign!")
                self.__update_global_plan(overtake_path=overtake_path)
                        
            if not self._overtake_manager.in_overtake and tw_distance < 6:
                print("[SCENARIO 3] - Emergency stop in front of the traffic warning sign!")
                return self.__emergency_stop()
        
        # If there is a construction cone, the agent will move of an offset to avoid it.
        elif not tw_state and cone_state and not self._overtake_manager._in_overtake:            
            # Get the location of the obstacle and the relative waypoint.
            o_loc = cone.get_location()
            o_wp = self._map.get_waypoint(o_loc)
            if o_wp.lane_id == ego_vehicle_wp.lane_id:
                print("[SCENARIO 3] - Cone is in the same lane of the ego vehicle!")
                self._local_planner.set_lateral_offset(-2 * cone.bounding_box.extent.y - self._vehicle.bounding_box.extent.y)
            else:
                print("[SCENARIO 3] - Cone is in the opposite lane of the ego vehicle!")
                self._local_planner.set_lateral_offset(.2 * cone.bounding_box.extent.y + self._vehicle.bounding_box.extent.y)

        # # If there are no obstacles, the agent will follow the lane.
        # else:
        #     self._local_planner.set_lateral_offset(0)
        
        # SCENARIO 4: Car behaviors
        vehicle_state, vehicle, vehicle_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        affected_by_stop_sign, _, sign_distance = self.stop_sign_manager(vehicle=self._vehicle, sign_distance=20)
        
        # SCENARIO 4.1 - Stop sign behavior (Stop sign is in the same lane of the ego vehicle - Junction behavior)
        if affected_by_stop_sign:
            # Distance is computed from the center of the two actors.
            sign_distance = compute_distance_from_center(actor1=self._vehicle, distance=sign_distance)
            print("[SCENARIO 4.1] - Stop sign at {0} meters away from the ego vehicle!".format(sign_distance))

            if sign_distance < 2 and not self._junction_manager.in_junction:
                print("--- Stopping at the stop sign!")
                # Set the speed limit to the maximum of the ego vehicle
                control = self._junction_manager.run_step(self._local_planner, look_ahead_steps=self._look_ahead_steps)
                        
                if control:
                    self.set_target_speed(self._speed_limit)
                    return self.__cross_junction()
                else:
                    return self.__emergency_stop()
            elif vehicle_state and sign_distance > 5: 
                print("--- Stop sign is far away! We must follow the vehicle in front of the ego vehicle.")
                # Follow the vehicle in front of the ego vehicle if it is in the same lane and the ego-vehicle is not too close to the stop sign.
                following_distance = compute_distance_from_center(actor1=self._vehicle, actor2=vehicle, distance=dist(self._vehicle, vehicle))
                return self.car_following_manager(vehicle=vehicle, distance=following_distance)
            else:
                print("--- Stop sign is far away! We must slow down.")
                self.set_target_speed(self._approach_speed)
                return self._local_planner.run_step()
       
        # SCENARIO 4.2 - Vehicle behaviors when it is in the same lane of the ego vehicle.
        if vehicle_state:
            vehicle_distance = compute_distance_from_center(actor1=self._vehicle, actor2=vehicle, distance=vehicle_distance)                   
            # Check if the vehicle is parked.
            vehicle_wp, parked = self._parked_vehicle(vehicle)
            
            print("[SCENARIO 4.2] Vehicle {0} is {1} meters away from the ego vehicle! It is parked? {2}".format(vehicle.id, vehicle_distance, parked))
             
            # SCENARIO 4.2.1 - Bicycle Logic
            if ego_vehicle_wp.lane_id == vehicle_wp.lane_id and is_a_bicycle(vehicle.type_id):
                # Get the yaw of the ego vehicle and the vehicle in front.
                ego_yaw = abs(self._vehicle.get_transform().rotation.yaw)
                vehicle_yaw = abs(vehicle.get_transform().rotation.yaw)

                if is_road_straight(ego_yaw=ego_yaw, vehicle_yaw=vehicle_yaw):                    
                    # Check if the bicycle is near the center of the lane. In this case, the agent will overtake the bicycle.
                    if is_bicycle_near_center(vehicle_location=vehicle.get_location(), ego_vehicle_wp=ego_vehicle_wp) and get_speed(self._vehicle) < 0.1:
                        print("--- Bicycle is near the center of the lane! We can try to overtake it.")
                        overtake_path = self._overtake_manager.run_step(
                            object_to_overtake=vehicle, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_from_object=vehicle_distance, speed_limit = self._speed_limit
                        )                                               
                        if overtake_path:
                            self.__update_global_plan(overtake_path=overtake_path)
                        if not self._overtake_manager.in_overtake:
                            return self.__emergency_stop()
                    # If the bicycle is not near the center of the lane, the agent will follow the lane. In particular, the agent will offset 
                    # the vehicle if the road is straight.
                    else:
                        print("--- Bicycle is not near the center of the lane! We can move of an offset to avoid it.")
                        self._local_planner.set_lateral_offset(-(2.5 * vehicle.bounding_box.extent.y + self._vehicle.bounding_box.extent.y))
                        control = self.__normal_behaviour(debug=debug)
                elif get_speed(vehicle) < 1:
                    # Set the target speed of the agent.
                    print("--- Bicycle is not moving! We decelerate.")
                    self.set_target_speed(self._approach_speed)
                    return self._local_planner.run_step()
                else:
                    print("--- Road is not straight! We follow it until the road is straight.")
                    control =  self.car_following_manager(vehicle, vehicle_distance, debug=debug)
            elif ego_vehicle_wp.lane_id != vehicle_wp.lane_id and ego_vehicle_wp == -1:
                    self._local_planner.set_lateral_offset(.2 * vehicle.bounding_box.extent.y + self._vehicle.bounding_box.extent.y)
                             
            # SCENARIO 4.2.3 - Overtake the vehicle in front of the ego vehicle if it is stopped and the ego vehicle is not in a junction.               
            elif vehicle_distance < self._behavior.braking_distance and not self._stuck:
                print("--- Vehicle is too close! We must brake.")
                self._stuck = True
                return self.__emergency_stop()
            
            # SCENARIO 4.2.2 - Overtake the vehicle in front of the ego vehicle if it is stopped and the ego vehicle is not in a junction.
            elif vehicle_wp.lane_id == ego_vehicle_wp.lane_id and parked and not ego_vehicle_wp.is_junction:               
                print("--- Vehicle is parked and not in a junction. We can try to overtake it.")
                overtake_path = self._overtake_manager.run_step(
                    object_to_overtake=vehicle, ego_vehicle_wp=ego_vehicle_wp, distance_same_lane=1, distance_from_object=vehicle_distance, speed_limit = self._speed_limit
                )            
                if overtake_path:
                    self.__update_global_plan(overtake_path=overtake_path)
                    
                if not self._overtake_manager.in_overtake:
                    return self.__emergency_stop()
                
                # If the agent is overtaking the vehicle, then it will follow the global plan.
                control = self.__normal_behaviour(debug=debug)
            
            # SCENARIO 4.2.4 - Vehicle is in the same lane of the ego vehicle but still distant from the ego vehicle. We must follow it.
            else:
                print("--- Vehicle is meters away from the ego vehicle! We must follow it.".format(vehicle, vehicle_distance))
                control = self.car_following_manager(vehicle, vehicle_distance, debug=debug)

        # SCENARIO 4.3 - Junction behaviors
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            print("[SCENARIO 4.3] - Junction behavior when the ego vehicle is in a junction but not affected by a stop sign.")
            target_speed = min([self._behavior.max_speed, self._speed_limit - 7])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
            
            if self._junction_right_cnt == 0:
                self._local_planner.set_lateral_offset(0)                
        
        # SCEANRIO 4.4 - Normal behavior
        else:
            print("[SCENARIO 4.4] - Car is following a normal behavior!")
            # Not change the lateral offset of the agent if there are obstacles. 
            # We can allow the agent of not changing the lateral offset if there are cones but not vehicles in the opposite lane.
            if not tw_state and not cone_state:
                self._local_planner.set_lateral_offset(0)
            self._stuck = False
            control = self.__normal_behaviour(debug=debug)

        return control
    
    # ====================================================================
    # -- Behavior Agent Public Functions ---------------------------------
    # ====================================================================
    def traffic_light_manager(self, vehicle : carla.Actor) -> bool:
        """
        This method is in charge of behaviors for red lights.
        
            :param vehicle (carla.Vehicle): vehicle object to be checked.
            
            :return affected (bool): True if the vehicle is affected by the red light, False otherwise.
        """
        affected, _ = self._affected_by_traffic_light(vehicle=vehicle)
        return affected
    
    def stop_sign_manager(self, vehicle : carla.Vehicle, sign_distance : int = 20) -> bool:
        """
        This method is in charge of behaviors for stop signs.
        
            :param vehicle (carla.Vehicle): vehicle object to be checked.
            :param sign_distance (int): distance to the stop sign.
            
            :return affected (bool): True if the vehicle is affected by the stop sign, False otherwise.
        """
        affected, signal = self._affected_by_sign(vehicle=vehicle, sign_type="206", max_distance=sign_distance)
        distance = -1 if not affected else dist(a=vehicle, b=signal)
        return affected, signal, distance

    def static_obstacle_manager(self, waypoint : carla.Waypoint, static_element="*static.prop*", angle_interval=[0, 90]):
        """
        This method is in charge of behaviors for handling static obstacles.
        
        args:
            - waypoint (carla.Waypoint): waypoint object.
            - static_element (str): static element to be detected.
        
        return:
            - bool: True if there is an obstacle, False otherwise.
            - carla.Actor: Obstacle object.
            - float: Distance to the obstacle.
        """
        # Get all the traffic warning signs in the simulation in order to avoid them during the navigation of the ego-vehicle.       
        obstacles_list = self._world.get_actors().filter(static_element)
        obstacles_list = [o for o in obstacles_list if is_within_distance(o.get_transform(), self._vehicle.get_transform(), 20, angle_interval=angle_interval)]
        
        # If there are no obstacles in the simulation, return False.
        if not obstacles_list:
            return False, None, -1
        
        # Sort the obstacles list based on the distance to the agent.
        obstacles_list = sorted(obstacles_list, key=lambda x: dist(x, waypoint))
        
        # Check if the agent is close to a traffic warning sign.
        if static_element == "*static.prop.constructioncone*":
            return True, obstacles_list[0], dist(obstacles_list[0], waypoint)
        
        # Check if the agent is close to a traffic warning sign.
        o_state, o, o_distance = self._vehicle_obstacle_detected(
            obstacles_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=60
        )
        
        return o_state, o, o_distance

    def _tailgating(self, waypoint : carla.Waypoint, vehicle_list : list) -> None:
        """
        This method is in charge of tailgating behaviors.

            :param waypoint (carla.Waypoint): waypoint object.
            :param vehicle_list (list): list of vehicles in the simulation.
        """
        # Look for the left and right lane changes of the waypoint
        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change
        # Get the left and right waypoints of the current waypoint
        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()
        
        speed_limit = self._vehicle.get_speed_limit()
        
        # Check if there is a vehicle behind the agent
        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, low_angle_th=160)
        
        # If there is a vehicle behind the agent and the agent is tailgating it, then change lanes
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            # Check if the agent can change lanes to the right
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                # Check if there is a vehicle in the right lane
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)
    
    
    def collision_and_car_avoid_manager(self, waypoint : carla.Waypoint) -> tuple:
        """
        This module is in charge of warning in case of a collision and managing possible tailgating chances.

            :param location (carla.Location): current location of the agent.
            :param waypoint (carla.Waypoint): current waypoint of the agent.
            
            :return vehicle_state (bool): True if there is a vehicle nearby, False if not.
            :return vehicle (carla.Vehicle): nearby vehicle to avoid.
            :return distance (float): distance to nearby vehicle.
        """
        # Get all the vehicles in the simulation in order to avoid them during the navigation of the ego-vehicle.
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        # Filter the vehicles that are too far away from the agent.
        vehicle_list = [v for v in vehicle_list if dist(v, waypoint) < 13 and v.id != self._vehicle.id]       

        # If there are no vehicles in the simulation, return False.
        if not vehicle_list:
            return False, None, -1

        # Get all the bicycles in the simulation in order to avoid them during the navigation of the ego-vehicle.
        bicycle_list = [b for b in vehicle_list if is_a_bicycle(b.type_id) and is_within_distance(b.get_transform(), self._vehicle.get_transform(), 10, angle_interval=[0, 90])]
                
        if len(bicycle_list) == 1:
            print('--- Bicycle crossing')
            return True, bicycle_list[0], dist(bicycle_list[0], waypoint)

        speed_limit = self._vehicle.get_speed_limit()
        # print("SPEED LIMIT: ", speed_limit)

        # Check if there is a vehicle in the proximity of the agent when the agent is changing lanes to the left.
        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, lane_offset=-1
            )
            
        # Check if there is a vehicle in the proximity of the agent when the agent is changing lanes to the right.
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, lane_offset=1
            )
            
        # Check if there is a vehicle in the proximity of the agent when the agent is following a lane.
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(self._behavior.min_proximity_threshold, speed_limit / 3), up_angle_th=30
            )

            if vehicle_state:
                vehicle_location = vehicle.get_location()
                vehicle_wp = self._map.get_waypoint(vehicle_location)
                if vehicle_wp.is_junction:
                    return vehicle_state, vehicle, distance

                projection_to_lane = vehicle_wp.get_left_lane()
                ego_wp = self._map.get_waypoint(self._vehicle.get_location())

                if projection_to_lane and projection_to_lane.lane_type == carla.LaneType.Driving and projection_to_lane.get_left_lane().lane_id == ego_wp.lane_id:
                    projection_to_lane = projection_to_lane.get_left_lane().transform.location
                    # compute the distance between the lane and the vehicle
                    distance_to_lane = vehicle_location.distance(projection_to_lane)
                    if distance_to_lane > self._vehicle.bounding_box.extent.y + vehicle.bounding_box.extent.y:
                        print('vehicle not in lane: ', vehicle, 'distance: ', distance_to_lane)
                        return False, None, -1

            # Check if the agent is tailgating another vehicle.
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:          
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint : carla.Waypoint) -> tuple:
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """
        # Get all the walkers in the simulation in order to avoid them during the navigation of the ego-vehicle.
        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        # Filter the walkers that are too far away from the agent.
        search_distance = 12 if self._junction_manager.in_junction else 20
        walker_list = [w for w in walker_list if is_within_distance(w.get_transform(), self._vehicle.get_transform(), search_distance, angle_interval=[0, 90])]
        # If there are no walkers in the simulation, return False.
        if not walker_list:
            return False, None, -1
        
        speed_limit = self._vehicle.get_speed_limit()
        # print("SPEED LIMIT: ", speed_limit)
        
        # If there is a walker in the proximity of the agent when the agent is in a junction, the agent will stop.
        if waypoint.is_junction:
            return True, walker_list[0], dist(walker_list[0], waypoint)  
                
        # Check if there is a walker in the proximity of the agent when the agent is changing lanes to the left.
        elif self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=90, lane_offset=-1)
        
        # Check if there is a walker in the proximity of the agent when the agent is changing lanes to the right.
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=90, lane_offset=1)
        
        # Check if there is a walker in the proximity of the agent when the agent is following a lane.
        else:            
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=90)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle : carla.Vehicle, distance : float, debug : bool = False) -> carla.VehicleControl:
        """
        Module in charge of car-following behaviors when there's someone in front of us.

            :param vehicle (carla.Vehicle): vehicle in front
            :param distance (float): distance to the vehicle in front
            :param debug (bool): debug flag to print information
            
            :return control (carla.VehicleControl): control to be applied to the agent
        """
        # Get the speed of the vehicle in front.
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    # ====================================================================
    # -- Behavior Agent Private Methods ----------------------------------
    # ====================================================================

    def __cross_junction(self, inhibite_junction : float = 120) -> carla.VehicleControl:
        """
        This function is called when the vehicle is ready to cross the junction. It returns the control to be applied at the junction.
        
            :param inhibite_junction (float): inhibite junction value to avoid the agent to stop at the junction.
        
            :return carla.VehicleControl: control to be applied at the junction.
        """
        self._junction_manager.in_junction_cnt = inhibite_junction
        self._junction_manager.in_junction = True
        if self._direction == RoadOption.RIGHT and self._junction_manager.junction_type == JunctionOption.FRONTLEFTRIGHT:
            self._junction_right_cnt = round(2 / self._world.get_snapshot().timestamp.delta_seconds)
            self._local_planner.set_lateral_offset(-0.3)
        return self._local_planner.run_step()

    def __normal_behaviour(self, debug : bool = False) -> carla.VehicleControl:
        """
        This method is in charge of the normal behavior of the agent. In particular, it is in charge of setting the speed of the agent and
        running the local planner.
        
            :param debug (bool): debug flag to print information.
        """
        # Set the speed of the agent.
        target_speed = min(
            [self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist]
        )
        self._local_planner.set_speed(target_speed)
        control = self._local_planner.run_step(debug=debug)
        return control
    
    def __emergency_stop(self) -> carla.VehicleControl:
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        # Get the vehicle control
        control = carla.VehicleControl()
        # Set the throttle and brake values to 0.0 and self._max_brake, respectively.
        control.throttle = 0.0
        control.brake = self._max_brake
        # Set the hand brake to False.
        control.hand_brake = False
        return control
    
    def __update_global_plan(self, overtake_path : list) -> None:
        """
        This method updates the global plan of the agent in order to overtake the vehicle in front of the agent.
        
            :param overtake_path (list): path to overtake the vehicle.
        """
        # Set the overtake plan and the target speed of the agent.
        new_plan = self._local_planner.set_overtake_plan(overtake_plan=overtake_path, overtake_distance=self._overtake_manager.overtake_ego_distance)
        # Update the speed of the agent in order to overtake the vehicle.
        self.set_target_speed(2 * self._speed_limit)
        # Set the global plan of the agent.
        self.set_global_plan(new_plan)
    
    def __update_information(self, debug : bool = False) -> None:
        """
        This method updates the information regarding the ego vehicle based on the surrounding world.
        
            :param debug (bool): debug flag to print information.
        """
        # Get the current weather of the simulation.
        self._weather = self._world.get_weather()
        precipitation_intensity = self._weather.precipitation
        preciptitation_deposits = self._weather.precipitation_deposits
        self._is_raining = precipitation_intensity > 50 or preciptitation_deposits > 55
                         
        # Update the speed of the agent.
        self._speed = get_speed(self._vehicle)

        # Update the speed limit of the agent.
        self._speed_limit = self._vehicle.get_speed_limit() 
        self._speed_limit -= 5 if self._is_raining else 0

        # Update the local planner speed.
        self._local_planner.set_speed(self._speed_limit)

        # Update the direction of the agent.
        self._direction = self._local_planner.target_road_option if self._local_planner.target_road_option is not None else RoadOption.LANEFOLLOW

        # Update the look ahead steps of the agent.
        self._look_ahead_steps = int(self._speed_limit / 10)

        # Update the incoming waypoint and direction of the agent.
        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps
        )

        # Update the incoming direction of the agent.
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW
            
        # Update the behavior of the agent.
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1       
        
        # Decrease the overtake counter if the agent is in the middle of an overtake maneuver.
        if self._overtake_manager.overtake_cnt > 0:
            self._overtake_manager.overtake_cnt -= 1
            self._local_planner.draw_waypoints(color=carla.Color(255, 125, 0))
        else: 
            self._overtake_manager.in_overtake = False
        
        # Decrease the junction counter if the agent is in the middle of a junction.
        if self._junction_manager.in_junction and self._junction_manager.in_junction_cnt <= 0:
            self._junction_manager.in_junction = False
            self._junction_manager.in_junction_cnt = 0
            self._junction_manager.stop_cnt = 0
        elif self._junction_manager.in_junction_cnt > 0:
            self._junction_manager.in_junction_cnt -= 1
            
        if self._junction_right_cnt > 0:
            self._junction_right_cnt -= 1
        
        if self._is_raining:
            print("[WORLD] It is raining!")
        print("[VEHICLE] Vehicle Speed: {0} - Speed Limit: {1}".format(self._speed, self._speed_limit))
                  
        # Print the information if debug is enabled.
        if debug:
            print("OVER TAKE COUNTER ", self._overtake_manager.overtake_cnt)
            print("IN OVERTAKE ", self._overtake_manager.in_overtake)
            print("IN JUNCTION COUNT: ", self._junction_manager.in_junction_cnt)