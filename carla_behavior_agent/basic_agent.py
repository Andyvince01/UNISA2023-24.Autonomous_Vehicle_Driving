# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
from shapely.geometry import Polygon

from local_planner import LocalPlanner, RoadOption
from global_route_planner import GlobalRoutePlanner
from misc import *

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0   # meters
        self._base_sign_threshold = 20.0    # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0
        self._simulation_timestamp = 0.05

        # Change parameters according to the dictionary
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']
        
        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation."""
        hazard_detected = False

        #####
        #  Retrieve all relevant actors
        #####
        # Basic Agent :
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        ### 

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._vehicle, self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True
            
        # Check if the vehicle is affected by a stop sign
        max_sign_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_sign, _ = self._affected_by_sign(self._vehicle, max_sign_distance, sign_type="206")
        if affected_by_sign:
            hazard_detected = True

        # Compute the local planner and retrieve the control signal
        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control
    
    def reset(self):
        pass

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=5):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)
    
    def _affected_by_traffic_light(self, vehicle : carla.Actor, lights_list : list = None, max_distance : float = None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
            
            :return (bool, carla.TrafficLight): a tuple containing a boolean indicating if the vehicle is affected by a traffic light
        """
        # If the agent is ignoring traffic lights, return False
        if self._ignore_traffic_lights:
            return (False, None)

        # If no list is provided, get all traffic lights
        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        # If no max distance is provided, use the default value
        if not max_distance:
            max_distance = self._base_tlight_threshold
            
        # If the traffic light is still the same, return it
        if self._last_traffic_light:
            # If the traffic light is not red anymore, reset it
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            # If the traffic light is still red, return it
            else:
                return (True, self._last_traffic_light)

        # Get the relevant information of the ego vehicle. We are interested in the metadata of the road. So, we extract them from its waypoint.
        # NOTE: A waypoint is a 3D-directed point in the map. Each waypoint contains a carla.Transform, which states its location on the map and 
        # the orientation of the line containing the waypoint. The variables road_id, section_id, and lane_id are the metadata of the road. 
        # Waypoints closer than 2 meters are considered to be in the same road, and so they have the same road_id.
        ego_vehicle_location = vehicle.get_location() 
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        # Check for traffic lights
        for traffic_light in lights_list:
            # Get the waypoint associated with this specific traffic light object from the map (if it exists)
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            # If the trigger location is not stored, get it
            else:
                # Get the trigger location of the traffic light. This is the point where the traffic light is placed.
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                # Store the trigger location
                self._lights_map[traffic_light.id] = trigger_wp

            # Check if the traffic light is relevant by checking the distance from the ego vehicle
            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            # Check if the traffic light is in the same road as the ego vehicle (that is, the road_id of the waypoint of the traffic light
            # is the same as the road_id of the waypoint of the ego vehicle)
            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            # Given the waypoint of the ego vehicle and the waypoint of the traffic light, we can calculate the direction of the road of 
            # the ego vehicle and the direction of the road of the traffic light.
            # NOTE: Given that the waypoint is a directed point, the direction of the road is given by the forward vector of the waypoint.
            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            # If the dot product is negative, the traffic light is behind the ego vehicle
            if dot_ve_wp < 0:
                continue

            # Check if the traffic light is RED
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            # Check if the traffic light is affecting the ego vehicle by checking the angle between the direction of the road of the ego vehicle.
            # NOTE: The angle between two vectors is given by the dot product of the two vectors divided by the product of the magnitudes of the vectors.
            if is_within_distance(trigger_wp.transform, vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _affected_by_sign(self, vehicle : carla.Vehicle, sign_type : str = "206", max_distance : float = None):
        '''
        This method checks if the vehicle is affected by a stop sign of a specific type.
        Default type is "206", which is the type of the stop sign.
        
            :param vehicle (carla.Vehicle): vehicle to check if it is affected by a stop sign
            :param sign_type (str): type of the stop sign
            :param max_distance (float): max distance for stop signs to be considered relevant.
            
            :return (bool, carla.Actor): a tuple containing a boolean indicating if the vehicle is affected by a stop sign
        '''
        
        if self._ignore_stop_signs:
            return False, None

        if max_distance is None:
            max_distance = self._base_sign_threshold

        # defines the last traffic light to which pay attention - case there is no sign registered
        target_vehicle_wp = self._map.get_waypoint(vehicle.get_location())
        signs_list = target_vehicle_wp.get_landmarks_of_type(max_distance, type=sign_type, stop_at_junction=False)

        # stop sign
        if sign_type == '206':
            signs_list = [sign for sign in signs_list if
                          self._map.get_waypoint(vehicle.get_location()).lane_id == sign.waypoint.lane_id]

            if signs_list:
                return True, signs_list[0]  # return type Landmark

        return False, None
        
        

    def _vehicle_obstacle_detected_old(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        
        # Nested function to get the route polygon to check for vehicles invading the lane due to the offset or junctions.
        def get_route_polygon():
            '''
            Function to get the polygon of the route to check for vehicles invading the lane due to the offset or junctions.
            
            :return: Polygon of the route
            '''
            # Initialize the route bounding box. It will be a list of points in the form [x, y, z]
            route_bb = []
            # Get the bounding box of the ego vehicle
            extent_y = self._vehicle.bounding_box.extent.y
            # Get the transform of the ego vehicle by extending the bounding box in the right and left directions by the offset value in order to get the route bounding box.
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            # Get the right vector of the ego vehicle
            r_vec = ego_transform.get_right_vector()
            # Get the points of the route bounding box by extending the bounding box of the ego vehicle in the right and left directions
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            # Add the points to the route bounding box
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Get the plan of the local planner and extend the route bounding box by adding the points of the plan
            for wp, _ in self._local_planner.get_plan():
                # If the distance between the ego vehicle and the waypoint is greater than the max distance, break the loop.
                # NOTE: This means that the route bounding box will be the route bounding box of the ego vehicle and the plan of the local 
                # planner up to the max distance. So, the route bounding box will be the route bounding box of the ego vehicle and the plan.
                if ego_location.distance(wp.transform.location) > max_distance:
                    break
                # Get the bounding box of the waypoint 
                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            # Return the polygon of the route
            return Polygon(route_bb)

        # If the agent is ignoring vehicles, return False
        if self._ignore_vehicles:
            return (False, None, -1)

        # If no list is provided, get all vehicles
        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        # If no max distance is provided, use the default value
        if not max_distance:
            max_distance = self._base_vehicle_threshold
            
        # Get the relevant information of the ego vehicle. We are interested in the metadata of the road. So, we extract them from its waypoint.
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset of the ego vehicle
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego vehicle by extending the bounding box in the forward direction
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        # Check for possible obstacles in the route bounding box
        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        # Check for obstacles in the scene by checking the distance between the ego vehicle and the other vehicles
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue
            
            # Get the transform of the target vehicle 
            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            # Get the waypoint of the target vehicle
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:
                # Get the bounding box of the target vehicle
                target_bb = target_vehicle.bounding_box
                # Get the vertices of the bounding box of the target vehicle
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                # Get the list of the vertices of the bounding box of the target vehicle
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                # Get the polygon of the bounding box of the target vehicle
                target_polygon = Polygon(target_list)

                # Check if the bounding box of the target vehicle intersects with the polygon of the route
                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:
                # Check if the target vehicle is in the same road and lane as the ego vehicle or in the lane offset
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    # Next waypoint in the route plan (3 steps ahead, that is, 6 meters ahead)
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]  
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue
                # Get the foward vector of the target vehicle (that is, the direction of the road of the target vehicle)
                target_forward_vector = target_transform.get_forward_vector()
                # Get the extent of the target vehicle (that is, the length of the target vehicle)
                target_extent = target_vehicle.bounding_box.extent.x
                # Get the transform of the rear of the target vehicle by extending the bounding box in the backward direction
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )
                
                # Check if the target vehicle is within the distance and angle thresholds of the ego vehicle
                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)
    
    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        # If the agent is ignoring vehicles, return False
        if self._ignore_vehicles:
            return (False, None, -1)

        # If no list is provided, get all vehicles
        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")
            

        # If no max distance is provided, use the default value
        if not max_distance:
            max_distance = self._base_vehicle_threshold

        # Get the relevant information of the ego vehicle. We are interested in the metadata of the road. So, we extract them from its waypoint.
        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset of the ego
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego vehicle by extending the bounding box in the forward direction
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        # Check for obstacles in the scene by checking the distance between the ego vehicle and the other vehicles
        for target_vehicle in vehicle_list:
            # Get the transform and the waypoint of the target vehicle
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # Get the route bounding box
            route_bb = []
            # Get the location of the ego vehicle
            ego_location = ego_transform.location
            # Get the extent of the bounding box of the ego vehicle
            extent_y = self._vehicle.bounding_box.extent.y
            # Get the right vector of the ego vehicle
            r_vec = ego_transform.get_right_vector()
            # Get the points of the route bounding box by extending the bounding box of the ego vehicle in the right and left directions
            p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
            p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
            route_bb.append([p1.x, p1.y, p1.z])
            route_bb.append([p2.x, p2.y, p2.z])

            # Get the plan of the local planner and extend the route bounding box by adding the points of the plan
            for wp, _ in self._local_planner.get_plan():
                # If the distance between the ego vehicle and the waypoint is greater than the max distance, break the loop.
                if ego_location.distance(wp.transform.location) > max_distance:
                    break
                # Get the bounding box of the waypoint in order to extend the route bounding box
                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return (False, None, -1)

            # Get the polygon of the route
            ego_polygon = Polygon(route_bb)

            # Compare the two polygons to check if the target vehicle is invading the lane of the ego vehicle
            for target_vehicle in vehicle_list:
                # Get the bounding box of the target vehicle in order to get the polygon of the target vehicle
                target_extent = target_vehicle.bounding_box.extent.x
                if target_vehicle.id == self._vehicle.id:
                    continue
                if ego_location.distance(target_vehicle.get_location()) > max_distance:
                    continue
                # Get the bounding box of the target vehicle
                target_bb = target_vehicle.bounding_box
                # Get the vertices of the bounding box of the target vehicle
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                # Get the list of the vertices of the bounding box of the target vehicle
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                # Get the polygon of the bounding box of the target vehicle
                target_polygon = Polygon(target_list)

                # Check if the polygon of the target vehicle intersects with the polygon of the route
                if ego_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))
                    
        # If no vehicle is detected, return False
        return (False, None, -1)

    def _generate_lane_change_path(self, 
                                   waypoint: carla.Waypoint, 
                                   direction: str = 'left',
                                   distance_same_lane: str = 10,
                                   distance_other_lane: int = 25, 
                                   lane_change_distance: int = 25,
                                   check: bool = True, 
                                   lane_changes: int = 1, 
                                   step_distance: float = 4.5,
                                   concorde: bool = False):  
        """
        This methods generates a path that results in a lane change. Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.

        params:
        - waypoint: starting waypoint of lane change
        - direction: overtake direction,'left' if overtake should be executed going left wrt waypoint, 'right' if going right
        - distance_same_lane: travelled distance in the same lane of 'waypoint'
        - distance_other_lane: travelled distance in the lane after the lane change
        - lane_change_distance: distance to move from 'waypoint' lane to the other', as horizontal component
        - check: flag to verify if the lane change is permitted on the current road
        - lane_changes: specifies how many lanes to move from 'waypoint' lane
        """
        # Initialize the plan with the current waypoint
        plan = [(waypoint, RoadOption.LANEFOLLOW)]
        # Initialize the option with the default value (LANEFOLLOW)
        option = RoadOption.LANEFOLLOW

        # 1. MOVE FORWARD IN THE CURRENT LANE
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # 2. CHANGE LANES
        while lane_changes_done < lane_changes:

            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Check if the lane change is permitted (if the road has a left lane)
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            # Check if the lane change is permitted (if the road has a right lane)
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            # Check if the lane change is permitted (if the road has a left or right lane)
            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # 3. MOVE FORWARD IN THE NEW LANE
        distance = 0
        pivot = plan[-1][0].lane_id
        while distance < distance_other_lane:
            if waypoint.lane_id * pivot > 0 and concorde:
                next_wps = plan[-1][0].next(step_distance)
            else:
                next_wps = plan[-1][0].previous(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan
       
    def _get_ordered_vehicles(self, reference : carla.Actor, max_distance: float) -> list:
        # Get all vehicles in the scene
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        # Sort the vehicles by distance to the vehicle in question
        # Remove the reference vehicle from the list
        if isinstance(reference, carla.Actor):
            vehicle_list = [
                v for v in vehicle_list
                if v.id != reference.id and 0.1 < dist(v, reference) < max_distance
            ]
        else:
            vehicle_list = [
                v for v in vehicle_list
                if 0.1 < dist(v, reference) < max_distance
            ]
        
        # Sort the vehicles by distance to the ego vehicle
        vehicle_list.sort(key=lambda v: dist(v, reference))
        return vehicle_list
    
    def _parked_vehicle(self, vehicle : carla.Vehicle) -> tuple:
        '''
        This method is in charge of checking if the vehicle in front of the ego vehicle is parked.
        It returns the waypoint of the vehicle in front of the ego vehicle and a boolean value that 
        indicates if the vehicle is parked.
        
            :param vehicle (carla.Vehicle): ego vehicle to check if the vehicle in front of it is parked.
            
            :return vehicle_wp (carla.Waypoint): waypoint of the vehicle in front of the ego vehicle.
            :return parked (bool): boolean value that indicates if the vehicle in front of the ego vehicle is parked.
        '''
        # Get vehicle location and waypoint.
        vehicle_loc = vehicle.get_location()
        vehicle_wp = self._map.get_waypoint(vehicle_loc)

        # Get the list of traffic lights in the scene near the ego vehicle.
        lights_list = self._world.get_actors().filter("*traffic_light*")
        lights_list = [l for l in lights_list if is_within_distance(l.get_transform(), vehicle.get_transform(), 50, angle_interval=[0, 90])]

        # Check if the vehicle is affected by a stop sign or by a traffic light.
        affected_by_traffic_light, _ = self._affected_by_traffic_light(self._vehicle)
        affected_by_stop_sign, _ = self._affected_by_sign(self._vehicle)

        if not affected_by_stop_sign or not affected_by_traffic_light:
            vehicle_list = self._get_ordered_vehicles(vehicle, 30)
            for v in vehicle_list:
                if v.id == self._vehicle.id:
                    continue
                affected_by_traffic_light, _ = self._affected_by_traffic_light(v)
                affected_by_stop_sign, _ = self._affected_by_sign(v)
                if affected_by_stop_sign or affected_by_traffic_light:
                    break

        if get_speed(vehicle) < 0.1 and not affected_by_stop_sign and not affected_by_traffic_light and not vehicle_wp.is_junction and not lights_list:
            return vehicle_wp, True
        
        return vehicle_wp, False