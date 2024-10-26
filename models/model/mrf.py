import numpy as np
import cv2
from collections import defaultdict
import math

class Node:
    def __init__(self, position, rgb, depth):
        self.position = position  # 像素位置 (x, y)
        self.rgb = rgb            # RGB值
        self.depth = depth        # 深度值
        self.neighbors = []       # 邻居节点列表
        self.unary_potential = None
        self.pairwise_potentials = {}
        
    def add_neighbor(self, neighbor_node):
        self.neighbors.append(neighbor_node)

def extract_lane_centers(predicted_lanes, image):
    """
    从预测结果中提取每条车道线的底部中心点坐标和RGB值。
    
    参数：
    - predicted_lanes: 预测的车道线列表，每条车道线由像素坐标组成的列表表示。
    - image: 输入的RGB图像。
    
    返回：
    - lane_centers: 包含每条车道线中心点信息的列表。
    """
    lane_centers = []
    height, width, _ = image.shape
    for lane in predicted_lanes:
        # 找到y坐标最大的点，即最底部的点
        bottom_point = max(lane, key=lambda p: p[1])
        x, y = int(bottom_point[0]), int(bottom_point[1])
        # 获取该点的RGB值
        if 0 <= x < width and 0 <= y < height:
            rgb = image[y, x, :]
        else:
            rgb = np.array([0, 0, 0])  # 超出图像范围，默认黑色
        lane_centers.append({'lane': lane, 'center_point': (x, y), 'rgb': rgb})
    return lane_centers

def construct_mrf_for_lane(lane_points, depth_map, image):
    """
    为一条车道线构建MRF模型。
    
    参数：
    - lane_points: 车道线的像素坐标列表。
    - depth_map: 深度图，与输入图像大小相同。
    - image: 输入的RGB图像。
    
    返回：
    - nodes: MRF模型的节点列表。
    """
    nodes = []
    height, width = depth_map.shape
    prev_node = None
    for idx, (x, y) in enumerate(lane_points):
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < width and 0 <= y_int < height:
            depth = depth_map[y_int, x_int]
            rgb = image[y_int, x_int, :]
            node = Node(position=(x_int, y_int), rgb=rgb, depth=depth)
            nodes.append(node)
            if prev_node:
                # 连接相邻节点
                prev_node.add_neighbor(node)
                node.add_neighbor(prev_node)
            prev_node = node
        else:
            # 如果坐标超出范围，可以选择跳过或进行外推
            continue
    return nodes

def compute_node_unary_potential(node, prev_node=None):
    """
    计算节点的Unary Potential，基于深度和RGB值的导数。
    
    参数：
    - node: 当前节点。
    - prev_node: 前一个节点。
    
    返回：
    - unary_potential: 节点的Unary Potential值。
    """
    if prev_node:
        # 计算深度导数
        depth_diff = abs(node.depth - prev_node.depth)
        depth_confidence = math.exp(-depth_diff)
        # 计算RGB导数
        rgb_diff = np.linalg.norm(node.rgb - prev_node.rgb)
        rgb_confidence = math.exp(-rgb_diff / 255.0)
        # 综合置信度
        unary_potential = depth_confidence * rgb_confidence
    else:
        # 起始节点，赋予默认置信度
        unary_potential = 1.0
    node.unary_potential = unary_potential
    return unary_potential

def compute_pairwise_potential(node1, node2):
    """
    计算两个相邻节点之间的Pairwise Potential。
    
    参数：
    - node1: 节点1。
    - node2: 节点2。
    
    返回：
    - pairwise_potential: 两个节点之间的Pairwise Potential值。
    """
    # 计算位置差异，鼓励空间上的连续性
    position_diff = np.linalg.norm(np.array(node1.position) - np.array(node2.position))
    spatial_confidence = math.exp(-position_diff)
    # 综合Potential
    pairwise_potential = spatial_confidence
    node1.pairwise_potentials[node2] = pairwise_potential
    node2.pairwise_potentials[node1] = pairwise_potential
    return pairwise_potential

def build_mrf_model(lane_centers, depth_map, image):
    """
    为所有车道线构建MRF模型。
    
    参数：
    - lane_centers: 包含每条车道线中心点信息的列表。
    - depth_map: 深度图。
    - image: 输入的RGB图像。
    
    返回：
    - mrf_models: 每条车道线对应的MRF模型列表。
    """
    mrf_models = []
    for lane_info in lane_centers:
        lane_points = lane_info['lane']
        nodes = construct_mrf_for_lane(lane_points, depth_map, image)
        # 计算Unary Potential和Pairwise Potential
        for idx, node in enumerate(nodes):
            if idx > 0:
                prev_node = nodes[idx - 1]
                compute_node_unary_potential(node, prev_node)
                compute_pairwise_potential(node, prev_node)
            else:
                # 起始节点
                compute_node_unary_potential(node)
        mrf_models.append(nodes)
    return mrf_models

def refine_lane_with_mrf(nodes, confidence_threshold=0.5):
    """
    使用MRF模型优化车道线，处理遮挡等情况。
    
    参数：
    - nodes: MRF模型的节点列表。
    - confidence_threshold: 置信度阈值，低于该值的节点将被调整。
    
    返回：
    - refined_lane: 优化后的车道线坐标列表。
    """
    refined_lane = []
    for node in nodes:
        if node.unary_potential >= confidence_threshold:
            refined_lane.append(node.position)
        else:
            # 对于低置信度节点，可以进行插值或使用邻居信息修正
            if node.neighbors:
                neighbor_positions = [neighbor.position for neighbor in node.neighbors if neighbor.unary_potential >= confidence_threshold]
                if neighbor_positions:
                    # 取邻居的平均位置
                    avg_position = np.mean(neighbor_positions, axis=0)
                    refined_lane.append(tuple(avg_position.astype(int)))
                else:
                    # 无法修正，跳过该点
                    continue
            else:
                continue
    return refined_lane

def process_lanes(predicted_lanes, depth_map, image):
    """
    主函数，处理预测的车道线，构建MRF模型并优化车道线。
    
    参数：
    - predicted_lanes: 预测的车道线列表。
    - depth_map: 深度图。
    - image: 输入的RGB图像。
    
    返回：
    - refined_lanes: 优化后的车道线列表。
    """
    # 提取车道线中心点
    lane_centers = extract_lane_centers(predicted_lanes, image)
    # 构建MRF模型
    mrf_models = build_mrf_model(lane_centers, depth_map, image)
    # 优化车道线
    refined_lanes = []
    for nodes in mrf_models:
        refined_lane = refine_lane_with_mrf(nodes)
        refined_lanes.append(refined_lane)
    return refined_lanes