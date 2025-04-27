from .base_dataset import BaseDataset
import torch
import numpy as np


class SimplDataset(BaseDataset):
    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # def pre_process(self, batch):
    #     """
    #     Preprocess the input batch data for SIMPL model.
        
    #     Args:
    #         batch: Dictionary containing input data
            
    #     Returns:
    #         Tuple containing:
    #         - actors: Tensor of actor features [batch_size, num_valid_objects, feature_dim, seq_len]
    #         - actor_idcs: Tensor of actor indices [batch_size, num_valid_objects]
    #         - lanes: Tensor of lane features [batch_size, num_valid_lanes, num_points, feature_dim]
    #         - lane_idcs: Tensor of lane indices [batch_size, num_valid_lanes]
    #         - rpe_prep: Dictionary containing relative position encoding data
    #         - actors_gt: Tensor of future actor states [batch_size, num_valid_objects, 4, future_len]
    #     """
    #     # Convert numpy arrays to tensors
    #     obj_trajs = torch.from_numpy(batch['obj_trajs']).to(self.device)  # [batch_size, N, T, 39]
    #     obj_trajs_mask = torch.from_numpy(batch['obj_trajs_mask']).to(self.device)  # [batch_size, N, T]
    #     obj_trajs_future = torch.from_numpy(batch['obj_trajs_future_state']).to(self.device)  # [batch_size, N, 60, 4]
    #     track_index_to_predict = torch.tensor(batch['track_index_to_predict'], device=self.device)
        
    #     # Get center object information
    #     center_objects_world = torch.tensor(batch['center_objects_world'], device=self.device)
    #     center_objects_type = torch.tensor(batch['center_objects_type'], device=self.device)
        
    #     # Extract map information
    #     map_polylines = torch.from_numpy(batch['map_polylines']).to(self.device)  # [batch_size, M, P, 29]
    #     map_polylines_mask = torch.from_numpy(batch['map_polylines_mask']).to(self.device)  # [batch_size, M, P]
        
    #     batch_size = obj_trajs.shape[0]
        
    #     # Process actors data
    #     valid_mask = obj_trajs_mask.any(dim=-1)  # [batch_size, num_objects]
    #     valid_objects = obj_trajs[valid_mask]  # [total_valid_objects, seq_len, 39]
    #     valid_future = obj_trajs_future[valid_mask]  # [total_valid_objects, 60, 4]
        
    #     # Extract position and heading
    #     trajs_pos = valid_objects[..., :3]  # [total_valid_objects, seq_len, 3]
    #     trajs_heading = valid_objects[..., 33:35]  # [total_valid_objects, seq_len, 2]
        
    #     # Get last time point position and heading for each object
    #     last_pos = trajs_pos[:, -1]  # [total_valid_objects, 3]
    #     last_heading = trajs_heading[:, -1]  # [total_valid_objects, 2]
        
    #     # Calculate rotation angles for each object
    #     thetas = torch.atan2(last_heading[:, 1], last_heading[:, 0])  # [total_valid_objects]
        
    #     # Create rotation matrices for each object
    #     cos_thetas = torch.cos(thetas)  # [total_valid_objects]
    #     sin_thetas = torch.sin(thetas)  # [total_valid_objects]
        
    #     # Create rotation matrices [total_valid_objects, 2, 2]
    #     rot_matrices = torch.stack([
    #         torch.stack([cos_thetas, -sin_thetas], dim=1),
    #         torch.stack([sin_thetas, cos_thetas], dim=1)
    #     ], dim=2)
        
    #     # First translate to origin (subtract last position)
    #     trajs_pos_centered = trajs_pos - last_pos.unsqueeze(1)  # [total_valid_objects, seq_len, 3]
        
    #     # Then rotate around origin
    #     # Reshape for batch matrix multiplication
    #     pos_reshaped = trajs_pos_centered[..., :2]  # [total_valid_objects, seq_len, 2]
    #     rot_reshaped = rot_matrices.unsqueeze(1)  # [total_valid_objects, 1, 2, 2]
        
    #     # Apply rotation to all trajectories at once
    #     pos_rotated = torch.matmul(pos_reshaped.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, seq_len, 2]
    #     trajs_pos_centered[..., :2] = pos_rotated
        
    #     # Transform velocities
    #     trajs_vel = valid_objects[..., 35:37]  # [total_valid_objects, seq_len, 2]
    #     vel_rotated = torch.matmul(trajs_vel.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, seq_len, 2]
    #     trajs_vel = vel_rotated
        
    #     # Transform headings
    #     heading_rotated = torch.matmul(trajs_heading.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, seq_len, 2]
    #     trajs_heading = heading_rotated
        
    #     # Extract object type
    #     trajs_type = valid_objects[..., 6:13]  # [total_valid_objects, seq_len, 7]
        
    #     # Get padding mask
    #     trajs_mask = obj_trajs_mask[valid_mask]  # [total_valid_objects, seq_len]
        
    #     # Combine features
    #     actor_features = torch.cat([
    #         trajs_pos_centered[..., :2],  # position (x, y)
    #         trajs_heading,  # heading encoding (sin, cos)
    #         trajs_vel,  # velocity (vx, vy)
    #         trajs_type,  # object type one-hot
    #         trajs_mask.unsqueeze(-1),  # padding mask
    #     ], dim=-1)  # [total_valid_objects, seq_len, feature_dim]
        
    #     # Exclude the last frame
    #     actor_features = actor_features[:, :-1, :]  # [total_valid_objects, 20, feature_dim]
    #     actors = actor_features.permute(0, 2, 1)  # [total_valid_objects, feature_dim, seq_len]
        
    #     # Process future trajectories
    #     future_pos = valid_future[..., :2]  # [total_valid_objects, 60, 2]
    #     future_vel = valid_future[..., 2:]  # [total_valid_objects, 60, 2]
        
    #     # Transform future positions
    #     future_pos_centered = future_pos - last_pos[:, :2].unsqueeze(1)  # [total_valid_objects, 60, 2]
    #     future_pos_rotated = torch.matmul(future_pos_centered.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, 60, 2]
        
    #     # Transform future velocities
    #     future_vel_rotated = torch.matmul(future_vel.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, 60, 2]
        
    #     # Combine future features
    #     actors_gt = torch.cat([
    #         future_pos_rotated,  # position (x, y)
    #         future_vel_rotated,  # velocity (vx, vy)
    #     ], dim=-1)  # [total_valid_objects, 60, 4]
        
    #     actors_gt = actors_gt.permute(0, 2, 1)  # [total_valid_objects, 4, 60]
        
    #     # Create actor indices
    #     actor_idcs = torch.arange(len(valid_objects), device=self.device)  # [total_valid_objects]
        
    #     # Process map data
    #     valid_lane_mask = map_polylines_mask.any(dim=-1)  # [batch_size, num_polylines]
    #     valid_polylines = map_polylines[valid_lane_mask]  # [total_valid_polylines, num_points, 29]
        
    #     # Extract position and direction
    #     lane_pos = valid_polylines[..., :3]  # [total_valid_polylines, num_points, 3]
    #     lane_dir = valid_polylines[..., 3:6]  # [total_valid_polylines, num_points, 3]
        
    #     # Calculate center point and heading for each lane
    #     lane_centers = lane_pos.mean(dim=1)  # [total_valid_polylines, 3]
    #     lane_vectors = lane_pos[:, -1] - lane_pos[:, 0]  # [total_valid_polylines, 3]
    #     lane_headings = torch.atan2(lane_vectors[:, 1], lane_vectors[:, 0])  # [total_valid_polylines]
        
    #     # Calculate rotation angles for each lane
    #     cos_thetas = torch.cos(lane_headings)  # [total_valid_polylines]
    #     sin_thetas = torch.sin(lane_headings)  # [total_valid_polylines]
        
    #     # Create rotation matrices [total_valid_polylines, 2, 2]
    #     rot_matrices = torch.stack([
    #         torch.stack([cos_thetas, -sin_thetas], dim=1),
    #         torch.stack([sin_thetas, cos_thetas], dim=1)
    #     ], dim=2)
        
    #     # First translate to center (subtract lane center)
    #     lane_pos_centered = lane_pos - lane_centers.unsqueeze(1)  # [total_valid_polylines, num_points, 3]
        
    #     # Then rotate around center
    #     # Reshape for batch matrix multiplication
    #     pos_reshaped = lane_pos_centered[..., :2]  # [total_valid_polylines, num_points, 2]
    #     rot_reshaped = rot_matrices.unsqueeze(1)  # [total_valid_polylines, 1, 2, 2]
        
    #     # Apply rotation to all lanes at once
    #     pos_rotated = torch.matmul(pos_reshaped.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_polylines, num_points, 2]
    #     lane_pos_centered[..., :2] = pos_rotated
        
    #     # Transform directions
    #     dir_rotated = torch.matmul(lane_dir[..., :2].unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_polylines, num_points, 2]
    #     lane_dir[..., :2] = dir_rotated
        
    #     # Extract lane type
    #     lane_type = valid_polylines[..., 9:29]  # [total_valid_polylines, num_points, 20]
        
    #     # Combine features
    #     lane_features = torch.cat([
    #         lane_pos_centered[..., :2],  # position (x, y)
    #         lane_dir[..., :2],  # direction (x, y)
    #         lane_type,  # lane type one-hot
    #     ], dim=-1)  # [total_valid_polylines, num_points, feature_dim]
        
    #     lanes = lane_features
        
    #     # Create lane indices
    #     lane_idcs = torch.arange(len(valid_polylines), device=self.device)  # [total_valid_polylines]
        
    #     # Prepare relative position encoding
    #     # Get original actor and lane features before localization
    #     actor_pos = valid_objects[..., :3]  # [total_valid_objects, seq_len, 3]
    #     actor_heading = valid_objects[..., 33:35]  # [total_valid_objects, seq_len, 2]
        
    #     # Get original lane positions and directions
    #     lane_pos = valid_polylines[..., :3]  # [total_valid_polylines, num_points, 3]
    #     lane_dir = valid_polylines[..., 3:6]  # [total_valid_polylines, num_points, 3]
        
    #     # Calculate relative positions and headings
    #     num_actors = actor_pos.shape[0]
    #     num_lanes = lane_pos.shape[0]
    #     total_nodes = num_actors + num_lanes
        
    #     # Initialize RPE tensors
    #     scene_rpe = torch.zeros(5, total_nodes, total_nodes, device=self.device)
    #     scene_mask = torch.zeros(total_nodes, total_nodes, device=self.device)
        
    #     # Calculate actor-actor RPE
    #     for i in range(num_actors):
    #         for j in range(num_actors):
    #             if i != j:
    #                 # Get last time point positions and headings
    #                 pos_i = actor_pos[i, -1]  # [3]
    #                 pos_j = actor_pos[j, -1]  # [3]
    #                 heading_i = actor_heading[i, -1]  # [2]
    #                 heading_j = actor_heading[j, -1]  # [2]
                    
    #                 # Relative position
    #                 rel_pos = pos_i[:2] - pos_j[:2]  # [2]
    #                 rel_dist = torch.norm(rel_pos)
                    
    #                 # Relative heading
    #                 heading_i_angle = torch.atan2(heading_i[1], heading_i[0])
    #                 heading_j_angle = torch.atan2(heading_j[1], heading_j[0])
    #                 rel_heading = heading_i_angle - heading_j_angle
    #                 rel_heading_sin = torch.sin(rel_heading)
    #                 rel_heading_cos = torch.cos(rel_heading)
                    
    #                 # Angle between heading and position vector
    #                 pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
    #                 angle_diff = pos_angle - heading_i_angle
    #                 angle_diff_sin = torch.sin(angle_diff)
    #                 angle_diff_cos = torch.cos(angle_diff)
                    
    #                 # Fill RPE
    #                 scene_rpe[0, i, j] = rel_heading_sin
    #                 scene_rpe[1, i, j] = rel_heading_cos
    #                 scene_rpe[2, i, j] = angle_diff_sin
    #                 scene_rpe[3, i, j] = angle_diff_cos
    #                 scene_rpe[4, i, j] = rel_dist
    #                 scene_mask[i, j] = 1
        
    #     # Calculate actor-lane RPE
    #     for i in range(num_actors):
    #         for j in range(num_lanes):
    #             # Get actor last time point position and heading
    #             pos_i = actor_pos[i, -1]  # [3]
    #             heading_i = actor_heading[i, -1]  # [2]
                
    #             # Get lane center and heading
    #             lane_center = lane_pos[j].mean(dim=0)  # [3]
    #             lane_vector = lane_pos[j, -1] - lane_pos[j, 0]  # [3]
    #             lane_heading = torch.atan2(lane_vector[1], lane_vector[0])
                
    #             # Relative position
    #             rel_pos = pos_i[:2] - lane_center[:2]  # [2]
    #             rel_dist = torch.norm(rel_pos)
                
    #             # Relative heading
    #             heading_i_angle = torch.atan2(heading_i[1], heading_i[0])
    #             rel_heading = heading_i_angle - lane_heading
    #             rel_heading_sin = torch.sin(rel_heading)
    #             rel_heading_cos = torch.cos(rel_heading)
                
    #             # Angle between heading and position vector
    #             pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
    #             angle_diff = pos_angle - heading_i_angle
    #             angle_diff_sin = torch.sin(angle_diff)
    #             angle_diff_cos = torch.cos(angle_diff)
                
    #             # Fill RPE
    #             scene_rpe[0, i, num_actors + j] = rel_heading_sin
    #             scene_rpe[1, i, num_actors + j] = rel_heading_cos
    #             scene_rpe[2, i, num_actors + j] = angle_diff_sin
    #             scene_rpe[3, i, num_actors + j] = angle_diff_cos
    #             scene_rpe[4, i, num_actors + j] = rel_dist
    #             scene_mask[i, num_actors + j] = 1
        
    #     # Calculate lane-lane RPE
    #     for i in range(num_lanes):
    #         for j in range(num_lanes):
    #             if i != j:
    #                 # Get lane centers and headings
    #                 lane_center_i = lane_pos[i].mean(dim=0)  # [3]
    #                 lane_center_j = lane_pos[j].mean(dim=0)  # [3]
    #                 lane_vector_i = lane_pos[i, -1] - lane_pos[i, 0]  # [3]
    #                 lane_vector_j = lane_pos[j, -1] - lane_pos[j, 0]  # [3]
    #                 lane_heading_i = torch.atan2(lane_vector_i[1], lane_vector_i[0])
    #                 lane_heading_j = torch.atan2(lane_vector_j[1], lane_vector_j[0])
                    
    #                 # Relative position
    #                 rel_pos = lane_center_i[:2] - lane_center_j[:2]  # [2]
    #                 rel_dist = torch.norm(rel_pos)
                    
    #                 # Relative heading
    #                 rel_heading = lane_heading_i - lane_heading_j
    #                 rel_heading_sin = torch.sin(rel_heading)
    #                 rel_heading_cos = torch.cos(rel_heading)
                    
    #                 # Angle between heading and position vector
    #                 pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
    #                 angle_diff = pos_angle - lane_heading_i
    #                 angle_diff_sin = torch.sin(angle_diff)
    #                 angle_diff_cos = torch.cos(angle_diff)
                    
    #                 # Fill RPE
    #                 scene_rpe[0, num_actors + i, num_actors + j] = rel_heading_sin
    #                 scene_rpe[1, num_actors + i, num_actors + j] = rel_heading_cos
    #                 scene_rpe[2, num_actors + i, num_actors + j] = angle_diff_sin
    #                 scene_rpe[3, num_actors + i, num_actors + j] = angle_diff_cos
    #                 scene_rpe[4, num_actors + i, num_actors + j] = rel_dist
    #                 scene_mask[num_actors + i, num_actors + j] = 1
        
    #     rpe_prep = {
    #         'scene': scene_rpe,  # [5, total_nodes, total_nodes]
    #         'scene_mask': scene_mask  # [total_nodes, total_nodes]
    #     }
        
    #     return actors, actor_idcs, lanes, lane_idcs, rpe_prep, actors_gt

    def pre_process(self, batch):
        """
        Preprocess the input batch data for SIMPL model.
        
        Args:
            batch: Dictionary containing input data
            
        Returns:
            Tuple containing:
            - actors: Tensor of actor features [batch_size, num_valid_objects, feature_dim, seq_len]
            - actor_idcs: Tensor of actor indices [batch_size, num_valid_objects]
            - lanes: Tensor of lane features [batch_size, num_valid_lanes, num_points, feature_dim]
            - lane_idcs: Tensor of lane indices [batch_size, num_valid_lanes]
            - rpe_prep: Dictionary containing relative position encoding data
            - actors_gt: Tensor of future actor states [batch_size, num_valid_objects, 4, future_len]
        """
        # Convert numpy arrays to tensors
        obj_trajs = torch.from_numpy(batch['obj_trajs']).to(self.device)  # [batch_size, N, T, 39]
        obj_trajs_mask = torch.from_numpy(batch['obj_trajs_mask']).to(self.device)  # [batch_size, N, T]
        obj_trajs_future = torch.from_numpy(batch['obj_trajs_future_state']).to(self.device)  # [batch_size, N, 60, 4]
        track_index_to_predict = torch.tensor(batch['track_index_to_predict'], device=self.device)
        
        # Get center object information
        center_objects_world = torch.tensor(batch['center_objects_world'], device=self.device)
        center_objects_type = torch.tensor(batch['center_objects_type'], device=self.device)
        
        # Extract map information
        map_polylines = torch.from_numpy(batch['map_polylines']).to(self.device)  # [batch_size, M, P, 29]
        map_polylines_mask = torch.from_numpy(batch['map_polylines_mask']).to(self.device)  # [batch_size, M, P]
        
        batch_size = obj_trajs.shape[0]
        
        # Process actors data
        valid_mask = obj_trajs_mask.any(dim=-1)  # [batch_size, num_objects]
        valid_objects = obj_trajs[valid_mask]  # [total_valid_objects, seq_len, 39]
        valid_future = obj_trajs_future[valid_mask]  # [total_valid_objects, 60, 4]
        
        # Extract position and heading
        trajs_pos = valid_objects[..., :3]  # [total_valid_objects, seq_len, 3]
        trajs_heading = valid_objects[..., 33:35]  # [total_valid_objects, seq_len, 2]
        
        # Get last time point position and heading for each object
        last_pos = trajs_pos[:, -1]  # [total_valid_objects, 3]
        last_heading = trajs_heading[:, -1]  # [total_valid_objects, 2]
        
        # Calculate rotation angles for each object
        thetas = torch.atan2(last_heading[:, 1], last_heading[:, 0])  # [total_valid_objects]
        
        # Create rotation matrices for each object
        cos_thetas = torch.cos(thetas)  # [total_valid_objects]
        sin_thetas = torch.sin(thetas)  # [total_valid_objects]
        
        # Create rotation matrices [total_valid_objects, 2, 2]
        rot_matrices = torch.stack([
            torch.stack([cos_thetas, -sin_thetas], dim=1),
            torch.stack([sin_thetas, cos_thetas], dim=1)
        ], dim=2)
        
        # First translate to origin (subtract last position)
        trajs_pos_centered = trajs_pos - last_pos.unsqueeze(1)  # [total_valid_objects, seq_len, 3]
        
        # Then rotate around origin
        # Reshape for batch matrix multiplication
        pos_reshaped = trajs_pos_centered[..., :2]  # [total_valid_objects, seq_len, 2]
        rot_reshaped = rot_matrices.unsqueeze(1)  # [total_valid_objects, 1, 2, 2]
        
        # Apply rotation to all trajectories at once
        pos_rotated = torch.matmul(pos_reshaped.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, seq_len, 2]
        trajs_pos_centered[..., :2] = pos_rotated
        
        # Transform velocities
        trajs_vel = valid_objects[..., 35:37]  # [total_valid_objects, seq_len, 2]
        vel_rotated = torch.matmul(trajs_vel.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, seq_len, 2]
        trajs_vel = vel_rotated
        
        # Transform headings
        heading_rotated = torch.matmul(trajs_heading.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, seq_len, 2]
        trajs_heading = heading_rotated
        
        # Extract object type
        trajs_type = valid_objects[..., 6:13]  # [total_valid_objects, seq_len, 7]
        
        # Get padding mask
        trajs_mask = obj_trajs_mask[valid_mask]  # [total_valid_objects, seq_len]
        
        # Combine features
        actor_features = torch.cat([
            trajs_pos_centered[..., :2],  # position (x, y)
            trajs_heading,  # heading encoding (sin, cos)
            trajs_vel,  # velocity (vx, vy)
            trajs_type,  # object type one-hot
            trajs_mask.unsqueeze(-1),  # padding mask
        ], dim=-1)  # [total_valid_objects, seq_len, feature_dim]
        
        # Exclude the last frame
        actor_features = actor_features[:, :-1, :]  # [total_valid_objects, 20, feature_dim]
        actors = actor_features.permute(0, 2, 1)  # [total_valid_objects, feature_dim, seq_len]
        
        # Process future trajectories
        future_pos = valid_future[..., :2]  # [total_valid_objects, 60, 2]
        future_vel = valid_future[..., 2:]  # [total_valid_objects, 60, 2]
        
        # Transform future positions
        future_pos_centered = future_pos - last_pos[:, :2].unsqueeze(1)  # [total_valid_objects, 60, 2]
        future_pos_rotated = torch.matmul(future_pos_centered.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, 60, 2]
        
        # Transform future velocities
        future_vel_rotated = torch.matmul(future_vel.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_objects, 60, 2]
        
        # Combine future features
        actors_gt = torch.cat([
            future_pos_rotated,  # position (x, y)
            future_vel_rotated,  # velocity (vx, vy)
        ], dim=-1)  # [total_valid_objects, 60, 4]
        
        actors_gt = actors_gt.permute(0, 2, 1)  # [total_valid_objects, 4, 60]
        
        # Create actor indices
        actor_idcs = torch.arange(len(valid_objects), device=self.device)  # [total_valid_objects]
        
        # Process map data
        valid_lane_mask = map_polylines_mask.any(dim=-1)  # [batch_size, num_polylines]
        valid_polylines = map_polylines[valid_lane_mask]  # [total_valid_polylines, num_points, 29]
        
        # Extract position and direction
        lane_pos = valid_polylines[..., :3]  # [total_valid_polylines, num_points, 3]
        lane_dir = valid_polylines[..., 3:6]  # [total_valid_polylines, num_points, 3]
        
        # Calculate center point and heading for each lane
        lane_centers = lane_pos.mean(dim=1)  # [total_valid_polylines, 3]
        lane_vectors = lane_pos[:, -1] - lane_pos[:, 0]  # [total_valid_polylines, 3]
        lane_headings = torch.atan2(lane_vectors[:, 1], lane_vectors[:, 0])  # [total_valid_polylines]
        
        # Calculate rotation angles for each lane
        cos_thetas = torch.cos(lane_headings)  # [total_valid_polylines]
        sin_thetas = torch.sin(lane_headings)  # [total_valid_polylines]
        
        # Create rotation matrices [total_valid_polylines, 2, 2]
        rot_matrices = torch.stack([
            torch.stack([cos_thetas, -sin_thetas], dim=1),
            torch.stack([sin_thetas, cos_thetas], dim=1)
        ], dim=2)
        
        # First translate to center (subtract lane center)
        lane_pos_centered = lane_pos - lane_centers.unsqueeze(1)  # [total_valid_polylines, num_points, 3]
        
        # Then rotate around center
        # Reshape for batch matrix multiplication
        pos_reshaped = lane_pos_centered[..., :2]  # [total_valid_polylines, num_points, 2]
        rot_reshaped = rot_matrices.unsqueeze(1)  # [total_valid_polylines, 1, 2, 2]
        
        # Apply rotation to all lanes at once
        pos_rotated = torch.matmul(pos_reshaped.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_polylines, num_points, 2]
        lane_pos_centered[..., :2] = pos_rotated
        
        # Transform directions
        dir_rotated = torch.matmul(lane_dir[..., :2].unsqueeze(-2), rot_reshaped).squeeze(-2)  # [total_valid_polylines, num_points, 2]
        lane_dir[..., :2] = dir_rotated
        
        # Extract lane type
        lane_type = valid_polylines[..., 9:29]  # [total_valid_polylines, num_points, 20]
        
        # Combine features
        lane_features = torch.cat([
            lane_pos_centered[..., :2],  # position (x, y)
            lane_dir[..., :2],  # direction (x, y)
            lane_type,  # lane type one-hot
        ], dim=-1)  # [total_valid_polylines, num_points, feature_dim]
        
        lanes = lane_features
        
        # Create lane indices
        lane_idcs = torch.arange(len(valid_polylines), device=self.device)  # [total_valid_polylines]
        
        # Prepare relative position encoding
        # Get original actor and lane features before localization
        actor_pos = valid_objects[..., :3]  # [total_valid_objects, seq_len, 3]
        actor_heading = valid_objects[..., 33:35]  # [total_valid_objects, seq_len, 2]
        
        # Get original lane positions and directions
        lane_pos = valid_polylines[..., :3]  # [total_valid_polylines, num_points, 3]
        lane_dir = valid_polylines[..., 3:6]  # [total_valid_polylines, num_points, 3]
        
        # Calculate relative positions and headings
        num_actors = actor_pos.shape[0]
        num_lanes = lane_pos.shape[0]
        total_nodes = num_actors + num_lanes
        
        # Initialize RPE tensors
        scene_rpe = torch.zeros(5, total_nodes, total_nodes, device=self.device)
        scene_mask = torch.zeros(total_nodes, total_nodes, device=self.device)
        
        # Calculate actor-actor RPE
        for i in range(num_actors):
            for j in range(num_actors):
                if i != j:
                    # Get last time point positions and headings
                    pos_i = actor_pos[i, -1]  # [3]
                    pos_j = actor_pos[j, -1]  # [3]
                    heading_i = actor_heading[i, -1]  # [2]
                    heading_j = actor_heading[j, -1]  # [2]
                    
                    # Relative position
                    rel_pos = pos_i[:2] - pos_j[:2]  # [2]
                    rel_dist = torch.norm(rel_pos)
                    
                    # Relative heading
                    heading_i_angle = torch.atan2(heading_i[1], heading_i[0])
                    heading_j_angle = torch.atan2(heading_j[1], heading_j[0])
                    rel_heading = heading_i_angle - heading_j_angle
                    rel_heading_sin = torch.sin(rel_heading)
                    rel_heading_cos = torch.cos(rel_heading)
                    
                    # Angle between heading and position vector
                    pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
                    angle_diff = pos_angle - heading_i_angle
                    angle_diff_sin = torch.sin(angle_diff)
                    angle_diff_cos = torch.cos(angle_diff)
                    
                    # Fill RPE
                    scene_rpe[0, i, j] = rel_heading_sin
                    scene_rpe[1, i, j] = rel_heading_cos
                    scene_rpe[2, i, j] = angle_diff_sin
                    scene_rpe[3, i, j] = angle_diff_cos
                    scene_rpe[4, i, j] = rel_dist
                    scene_mask[i, j] = 1
        
        # Calculate actor-lane RPE
        for i in range(num_actors):
            for j in range(num_lanes):
                # Get actor last time point position and heading
                pos_i = actor_pos[i, -1]  # [3]
                heading_i = actor_heading[i, -1]  # [2]
                
                # Get lane center and heading
                lane_center = lane_pos[j].mean(dim=0)  # [3]
                lane_vector = lane_pos[j, -1] - lane_pos[j, 0]  # [3]
                lane_heading = torch.atan2(lane_vector[1], lane_vector[0])
                
                # Relative position
                rel_pos = pos_i[:2] - lane_center[:2]  # [2]
                rel_dist = torch.norm(rel_pos)
                
                # Relative heading
                heading_i_angle = torch.atan2(heading_i[1], heading_i[0])
                rel_heading = heading_i_angle - lane_heading
                rel_heading_sin = torch.sin(rel_heading)
                rel_heading_cos = torch.cos(rel_heading)
                
                # Angle between heading and position vector
                pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
                angle_diff = pos_angle - heading_i_angle
                angle_diff_sin = torch.sin(angle_diff)
                angle_diff_cos = torch.cos(angle_diff)
                
                # Fill RPE
                scene_rpe[0, i, num_actors + j] = rel_heading_sin
                scene_rpe[1, i, num_actors + j] = rel_heading_cos
                scene_rpe[2, i, num_actors + j] = angle_diff_sin
                scene_rpe[3, i, num_actors + j] = angle_diff_cos
                scene_rpe[4, i, num_actors + j] = rel_dist
                scene_mask[i, num_actors + j] = 1
        
        # Calculate lane-lane RPE
        for i in range(num_lanes):
            for j in range(num_lanes):
                if i != j:
                    # Get lane centers and headings
                    lane_center_i = lane_pos[i].mean(dim=0)  # [3]
                    lane_center_j = lane_pos[j].mean(dim=0)  # [3]
                    lane_vector_i = lane_pos[i, -1] - lane_pos[i, 0]  # [3]
                    lane_vector_j = lane_pos[j, -1] - lane_pos[j, 0]  # [3]
                    lane_heading_i = torch.atan2(lane_vector_i[1], lane_vector_i[0])
                    lane_heading_j = torch.atan2(lane_vector_j[1], lane_vector_j[0])
                    
                    # Relative position
                    rel_pos = lane_center_i[:2] - lane_center_j[:2]  # [2]
                    rel_dist = torch.norm(rel_pos)
                    
                    # Relative heading
                    rel_heading = lane_heading_i - lane_heading_j
                    rel_heading_sin = torch.sin(rel_heading)
                    rel_heading_cos = torch.cos(rel_heading)
                    
                    # Angle between heading and position vector
                    pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
                    angle_diff = pos_angle - lane_heading_i
                    angle_diff_sin = torch.sin(angle_diff)
                    angle_diff_cos = torch.cos(angle_diff)
                    
                    # Fill RPE
                    scene_rpe[0, num_actors + i, num_actors + j] = rel_heading_sin
                    scene_rpe[1, num_actors + i, num_actors + j] = rel_heading_cos
                    scene_rpe[2, num_actors + i, num_actors + j] = angle_diff_sin
                    scene_rpe[3, num_actors + i, num_actors + j] = angle_diff_cos
                    scene_rpe[4, num_actors + i, num_actors + j] = rel_dist
                    scene_mask[num_actors + i, num_actors + j] = 1
        
        rpe_prep = {
            'scene': scene_rpe,  # [5, total_nodes, total_nodes]
            'scene_mask': scene_mask  # [total_nodes, total_nodes]
        }
        
        return actors, actor_idcs, lanes, lane_idcs, rpe_prep, actors_gt

    def __getitem__(self, idx):
        # Get raw data from BaseDataset
        batch = super().__getitem__(idx)
        
        # Pre-process data
        actors, actor_idcs, lanes, lane_idcs, rpe_prep, actors_gt = self.pre_process(batch)
        
        # Ensure all outputs are tensors but keep them on CPU
        if not isinstance(actors, torch.Tensor):
            actors = torch.tensor(actors)  # device 지정 제거
        if not isinstance(actor_idcs, torch.Tensor):
            actor_idcs = torch.tensor(actor_idcs)  # device 지정 제거
        if not isinstance(lanes, torch.Tensor):
            lanes = torch.tensor(lanes)  # device 지정 제거
        if not isinstance(lane_idcs, torch.Tensor):
            lane_idcs = torch.tensor(lane_idcs)  # device 지정 제거
        
        return {
            'actors': actors,
            'actor_idcs': actor_idcs,
            'lanes': lanes,
            'lane_idcs': lane_idcs,
            'rpe_prep': rpe_prep,
            'actors_gt': actors_gt,
            'batch': batch  # Keep original batch for other information
        }

    def collate_fn(self, data_list):
        # 1. Collect all batches
        batch_list = []
        for batch in data_list:
            batch_list.append(batch)

        # 2. Get batch size
        batch_size = len(batch_list)

        # 3. Organize data by keys
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        # 4. Convert lists to tensors
        input_dict = {}
        for key, val_list in key_to_list.items():
            if key == 'rpe_prep':
                # rpe_prep는 리스트 형태 유지, 각 요소는 {'scene': tensor, 'scene_mask': tensor}
                input_dict[key] = val_list  # 리스트 형태 그대로 유지
            elif key in ['actors', 'lanes', 'actors_gt']:
                # actors, lanes, actors_gt는 모든 배치의 텐서를 하나로 합침
                if isinstance(val_list[0], torch.Tensor):
                    input_dict[key] = torch.cat(val_list, dim=0)  # [total_num_valid_objects, ...]
                else:
                    input_dict[key] = torch.from_numpy(np.concatenate(val_list, axis=0))
            elif key == 'actor_idcs':
                # Create new indices for actors
                actor_idcs = []
                current_idx = 0
                for batch in val_list:
                    num_actors = len(batch)
                    new_indices = torch.arange(current_idx, current_idx + num_actors, device=self.device)
                    actor_idcs.append(new_indices)
                    current_idx += num_actors
                input_dict[key] = actor_idcs
            elif key == 'lane_idcs':
                # Create new indices for lanes
                lane_idcs = []
                current_idx = 0
                for batch in val_list:
                    num_lanes = len(batch)
                    new_indices = torch.arange(current_idx, current_idx + num_lanes, device=self.device)
                    lane_idcs.append(new_indices)
                    current_idx += num_lanes
                input_dict[key] = lane_idcs
            else:
                # 나머지 데이터는 기본 처리
                try:
                    if isinstance(val_list[0], torch.Tensor):
                        input_dict[key] = torch.stack(val_list, dim=0)
                    else:
                        input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
                except:
                    input_dict[key] = val_list

        # 5. Create final batch dictionary
        batch_dict = {
            'batch_size': batch_size,
            'input_dict': input_dict,
            'batch_sample_count': batch_size
        }

        return batch_dict
    