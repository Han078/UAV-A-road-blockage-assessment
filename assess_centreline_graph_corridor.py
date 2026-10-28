#!/usr/bin/env python3
"""Centreline Graph Corridor V1.3 standalone, concise-comment edition."""

from __future__ import annotations

import sys
import types


def _load_embedded_module(name: str, source: str) -> types.ModuleType:
    """Load one bundled source section as an in-memory module."""
    module = types.ModuleType(name)
    module.__file__ = __file__
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, f"<embedded:{name}>", "exec"), module.__dict__)
    return module


_RECONSTRUCT_ROAD_V1_SOURCE = r'''
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Iterable, Optional, Sequence
try:
    import cv2
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import distance_transform_edt, median_filter
    from scipy.signal import savgol_filter
    from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects, skeletonize
except ImportError as exc:
    raise SystemExit(f'缺少套件。請先執行：\npip install numpy opencv-python scipy scikit-image\n原始錯誤：{exc}') from exc
OBSTACLE_LABELS = {'debris pile', 'fallen vegetation', 'rock'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='從 road GT polygon 重建空拍影像中的假想完整道路（第一版）')
    parser.add_argument('--json', required=True, type=Path, help='彙整後的標註 JSON')
    parser.add_argument('--images-dir', type=Path, default=None, help='原始圖片資料夾（可省略）')
    parser.add_argument('--output-dir', required=True, type=Path, help='輸出資料夾')
    parser.add_argument('--only', default=None, help='只處理指定 key/檔名（可用完整檔名或不含副檔名的 stem）')
    parser.add_argument('--max-images', type=int, default=None, help='最多處理幾張')
    parser.add_argument('--sample-step', type=float, default=4.0, help='中心線採樣間距（pixel）')
    parser.add_argument('--min-road-area', type=int, default=300, help='移除小於此面積的 road 區塊')
    parser.add_argument('--min-centerline-length', type=int, default=30, help='忽略過短的骨架分支')
    parser.add_argument('--closing-radius', type=int, default=2, help='road mask 形態學閉運算半徑')
    parser.add_argument('--gap-max-widths', type=float, default=12.0, help='允許補接的最大斷點距離，以道路中位寬度倍數表示')
    parser.add_argument('--cross-section-stride', type=int, default=20, help='診斷圖每隔多少個中心線採樣點畫一條橫斷面；0 表示不畫')
    return parser.parse_args()

def load_items(path: Path) -> list[dict]:
    """Load annotation items from JSON."""
    with path.open('r', encoding='utf-8-sig') as file:
        data = json.load(file)
    if isinstance(data, dict) and isinstance(data.get('items'), list):
        return data['items']
    if isinstance(data, list):
        return data
    raise ValueError('JSON 必須是含有 items 陣列的彙整格式，或直接是一個 item 陣列。')

def polygon_from_box(box: dict) -> Optional[np.ndarray]:
    """Polygon from box."""
    points = box.get('points')
    if isinstance(points, list) and len(points) >= 3:
        if isinstance(points[0], dict):
            array = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        elif isinstance(points[0], (list, tuple)) and len(points[0]) >= 2:
            array = np.asarray(points, dtype=np.float32)[:, :2]
        else:
            array = np.empty((0, 2), dtype=np.float32)
        if len(array) >= 3:
            return np.rint(array).astype(np.int32)
    required = ('x', 'y', 'width', 'height')
    if all((box.get(key) is not None for key in required)):
        x, y = (float(box['x']), float(box['y']))
        width, height = (float(box['width']), float(box['height']))
        return np.rint(np.array([[x - width / 2, y - height / 2], [x + width / 2, y - height / 2], [x + width / 2, y + height / 2], [x - width / 2, y + height / 2]])).astype(np.int32)
    return None

def rasterize_labels(item: dict, labels: set[str]) -> np.ndarray:
    """Convert selected polygon labels into a binary mask."""
    height, width = (int(item['height']), int(item['width']))
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in item.get('boxes', []):
        if str(box.get('label', '')).strip().lower() not in labels:
            continue
        polygon = polygon_from_box(box)
        if polygon is None:
            continue
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)

def clean_road_mask(road: np.ndarray, min_area: int, closing_radius: int) -> np.ndarray:
    """Clean small regions and holes in the road mask."""
    cleaned = remove_small_objects(road.astype(bool), min_size=max(1, min_area))
    cleaned = remove_small_holes(cleaned, area_threshold=max(1, min_area))
    if closing_radius > 0:
        cleaned = binary_closing(cleaned, footprint=disk(closing_radius))
    return cleaned.astype(bool)
NEIGHBOURS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

def skeleton_components(skeleton: np.ndarray) -> list[np.ndarray]:
    """Skeleton components."""
    count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    return [np.argwhere(labels == index) for index in range(1, count)]

def component_neighbours(node: tuple[int, int], node_set: set[tuple[int, int]]) -> Iterable[tuple[int, int]]:
    """Component neighbours."""
    y, x = node
    for dy, dx in NEIGHBOURS_8:
        neighbour = (y + dy, x + dx)
        if neighbour in node_set:
            yield neighbour

def bfs_farthest(start: tuple[int, int], node_set: set[tuple[int, int]]) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], int]]:
    """Bfs farthest."""
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    distance = {start: 0}
    farthest = start
    while queue:
        current = queue.popleft()
        if distance[current] > distance[farthest]:
            farthest = current
        for neighbour in component_neighbours(current, node_set):
            if neighbour not in distance:
                distance[neighbour] = distance[current] + 1
                parent[neighbour] = current
                queue.append(neighbour)
    return (farthest, parent, distance)

def longest_ordered_path(component_yx: np.ndarray) -> np.ndarray:
    """Longest ordered path."""
    nodes = {tuple(map(int, point)) for point in component_yx}
    if not nodes:
        return np.empty((0, 2), dtype=np.float32)
    endpoints = [node for node in nodes if sum((1 for _ in component_neighbours(node, nodes))) == 1]
    start = endpoints[0] if endpoints else next(iter(nodes))
    first, _, _ = bfs_farthest(start, nodes)
    second, parent, _ = bfs_farthest(first, nodes)
    path_yx = [second]
    while path_yx[-1] != first:
        path_yx.append(parent[path_yx[-1]])
    path_yx.reverse()
    return np.array([[x, y] for y, x in path_yx], dtype=np.float32)

def extract_centerline_paths(road: np.ndarray, min_length: int) -> list[np.ndarray]:
    """Extract ordered centreline paths from a road mask."""
    skeleton = skeletonize(road)
    paths = []
    for component in skeleton_components(skeleton):
        path = longest_ordered_path(component)
        if len(path) >= min_length:
            paths.append(path)
    return sorted(paths, key=len, reverse=True)

def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize."""
    norm = float(np.linalg.norm(vector))
    if norm < 1e-06:
        return np.zeros(2, dtype=np.float32)
    return (vector / norm).astype(np.float32)

def endpoint_tangent(path: np.ndarray, at_start: bool, lookahead: int=10) -> np.ndarray:
    """Endpoint tangent."""
    offset = min(lookahead, len(path) - 1)
    if at_start:
        return normalize(path[offset] - path[0])
    return normalize(path[-1] - path[-1 - offset])

def orient_path_for_connection(path: np.ndarray, selected_end: int, as_left: bool) -> np.ndarray:
    """Orient path for connection."""
    if as_left:
        return path if selected_end == 1 else path[::-1].copy()
    return path if selected_end == 0 else path[::-1].copy()

def connection_score(left: np.ndarray, left_end: int, right: np.ndarray, right_end: int, scale: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Connection score."""
    a = orient_path_for_connection(left, left_end, as_left=True)
    b = orient_path_for_connection(right, right_end, as_left=False)
    chord_vector = b[0] - a[-1]
    gap = float(np.linalg.norm(chord_vector))
    chord = normalize(chord_vector)
    tangent_a = endpoint_tangent(a, at_start=False)
    tangent_b = endpoint_tangent(b, at_start=True)
    alignment_penalty = (1.0 - float(np.dot(tangent_a, chord))) / 2.0
    alignment_penalty += (1.0 - float(np.dot(tangent_b, chord))) / 2.0
    score = gap / max(scale, 1.0) + 2.5 * alignment_penalty
    return (score, gap, a, b)

def hermite_bridge(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    """Hermite bridge."""
    p0, p1 = (a[-1], b[0])
    gap = float(np.linalg.norm(p1 - p0))
    sample_count = max(3, int(math.ceil(gap / max(step, 1.0))) + 1)
    t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)[:, None]
    tangent_scale = min(gap * 0.75, max(gap * 0.35, 4.0 * step))
    m0 = endpoint_tangent(a, at_start=False) * tangent_scale
    m1 = endpoint_tangent(b, at_start=True) * tangent_scale
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return (h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1).astype(np.float32)

def join_centerline_paths(paths: list[np.ndarray], median_width: float, max_gap_widths: float, step: float) -> tuple[np.ndarray, int, list[float]]:
    """Join centerline paths."""
    if not paths:
        return (np.empty((0, 2), dtype=np.float32), 0, [])
    work = [path.copy() for path in paths]
    bridges = 0
    gap_lengths: list[float] = []
    maximum_gap = max(30.0, max_gap_widths * max(median_width, 1.0))
    while len(work) > 1:
        best = None
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                for end_i in (0, 1):
                    for end_j in (0, 1):
                        score, gap, a, b = connection_score(work[i], end_i, work[j], end_j, median_width)
                        if gap <= maximum_gap and (best is None or score < best[0]):
                            best = (score, gap, i, j, a, b)
        if best is None:
            break
        _, gap, i, j, a, b = best
        bridge = hermite_bridge(a, b, step)
        joined = np.vstack([a, bridge[1:-1], b])
        work[i] = joined
        del work[j]
        bridges += 1
        gap_lengths.append(gap)
    path = max(work, key=polyline_length)
    return (path, bridges, gap_lengths)

def polyline_length(path: np.ndarray) -> float:
    """Polyline length."""
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

def resample_and_smooth(path: np.ndarray, step: float) -> np.ndarray:
    """Resample and smooth."""
    if len(path) < 2:
        return path.astype(np.float32)
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-06]
    path = path[keep]
    if len(path) < 2:
        return path.astype(np.float32)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    positions = np.arange(0.0, distance[-1] + step, step)
    positions[-1] = distance[-1]
    sampled = np.column_stack([np.interp(positions, distance, path[:, axis]) for axis in (0, 1)]).astype(np.float32)
    if len(sampled) >= 7:
        window = min(21, len(sampled) if len(sampled) % 2 else len(sampled) - 1)
        if window >= 5:
            sampled[:, 0] = savgol_filter(sampled[:, 0], window, 2, mode='interp')
            sampled[:, 1] = savgol_filter(sampled[:, 1], window, 2, mode='interp')
    return sampled

def path_normals(path: np.ndarray) -> np.ndarray:
    """Path normals."""
    if len(path) < 2:
        return np.zeros_like(path)
    tangent = np.gradient(path, axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norms, 1e-06)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]]).astype(np.float32)

def sample_mask(mask: np.ndarray, point_xy: np.ndarray) -> bool:
    """Sample mask."""
    x, y = np.rint(point_xy).astype(int)
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])

def ray_distance(mask: np.ndarray, point: np.ndarray, direction: np.ndarray, maximum: float) -> float:
    """Ray distance."""
    last_inside = 0.0
    distance = 0.5
    while distance <= maximum:
        if not sample_mask(mask, point + direction * distance):
            return last_inside
        last_inside = distance
        distance += 0.5
    return last_inside

def obstacle_near_point(obstacle_distance: np.ndarray, point: np.ndarray, clearance: float) -> bool:
    """Obstacle near point."""
    x, y = np.rint(point).astype(int)
    if not (0 <= x < obstacle_distance.shape[1] and 0 <= y < obstacle_distance.shape[0]):
        return True
    return bool(obstacle_distance[y, x] <= clearance)

def measure_and_interpolate_widths(road: np.ndarray, obstacle: np.ndarray, path: np.ndarray, median_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure and interpolate widths."""
    normals = path_normals(path)
    left = np.full(len(path), np.nan, dtype=np.float32)
    right = np.full(len(path), np.nan, dtype=np.float32)
    obstacle_distance = distance_transform_edt(~obstacle)
    ray_maximum = max(40.0, 2.5 * median_width)
    minimum_width = max(3.0, 0.35 * median_width)
    maximum_width = max(12.0, 2.5 * median_width)
    for index, (point, normal) in enumerate(zip(path, normals)):
        if not sample_mask(road, point):
            continue
        if obstacle.any() and obstacle_near_point(obstacle_distance, point, clearance=0.3 * median_width):
            continue
        left_value = ray_distance(road, point, normal, ray_maximum)
        right_value = ray_distance(road, point, -normal, ray_maximum)
        total = left_value + right_value
        if minimum_width <= total <= maximum_width and left_value > 0 and (right_value > 0):
            left[index], right[index] = (left_value, right_value)
    valid = np.isfinite(left) & np.isfinite(right)
    fallback = max(2.0, median_width / 2.0)

    def interpolate(values: np.ndarray) -> np.ndarray:
        """Interpolate."""
        indices = np.arange(len(values), dtype=np.float32)
        valid_indices = indices[valid]
        if len(valid_indices) == 0:
            result = np.full(len(values), fallback, dtype=np.float32)
        elif len(valid_indices) == 1:
            result = np.full(len(values), values[valid][0], dtype=np.float32)
        else:
            observed = median_filter(values[valid], size=min(5, len(valid_indices)))
            result = PchipInterpolator(valid_indices, observed, extrapolate=False)(indices)
            result = np.where(np.isfinite(result), result, np.interp(indices, valid_indices, observed))
        return np.clip(result, 0.15 * median_width, 1.35 * median_width).astype(np.float32)
    return (interpolate(left), interpolate(right), valid)

def sweep_road_polygon(shape: tuple[int, int], path: np.ndarray, left_width: np.ndarray, right_width: np.ndarray) -> np.ndarray:
    """Sweep road polygon."""
    result = np.zeros(shape, dtype=np.uint8)
    if len(path) < 2:
        return result.astype(bool)
    normals = path_normals(path)
    left_edge = path + normals * left_width[:, None]
    right_edge = path - normals * right_width[:, None]
    polygon = np.vstack([left_edge, right_edge[::-1]])
    polygon[:, 0] = np.clip(polygon[:, 0], 0, shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, shape[0] - 1)
    cv2.fillPoly(result, [np.rint(polygon).astype(np.int32)], 1)
    return result.astype(bool)

def estimate_median_width(road: np.ndarray, paths: Sequence[np.ndarray]) -> float:
    """Estimate median width."""
    distance = distance_transform_edt(road)
    values = []
    for path in paths:
        xy = np.rint(path).astype(int)
        valid = (xy[:, 0] >= 0) & (xy[:, 0] < road.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < road.shape[0])
        xy = xy[valid]
        if len(xy):
            radii = distance[xy[:, 1], xy[:, 0]]
            values.extend((2.0 * radii[radii > 1.0]).tolist())
    return float(np.median(values)) if values else 20.0

def unicode_imread(path: Path) -> Optional[np.ndarray]:
    """Unicode imread."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None

def unicode_imwrite(path: Path, image: np.ndarray) -> None:
    """Unicode imwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or '.png'
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f'無法編碼影像：{path}')
    encoded.tofile(str(path))

def find_image(images_dir: Optional[Path], key: str) -> Optional[Path]:
    """Find image."""
    if images_dir is None:
        return None
    exact = images_dir / key
    if exact.is_file():
        return exact
    stem = Path(key).stem
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f'{stem}{extension}'
        if candidate.is_file():
            return candidate
    for candidate in images_dir.rglob(f'{stem}.*'):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None

def make_overlay(image: Optional[np.ndarray], road_visible: np.ndarray, road_reconstructed: np.ndarray, obstacle: np.ndarray, path: np.ndarray, left_width: np.ndarray, right_width: np.ndarray, cross_section_stride: int) -> np.ndarray:
    """Draw masks, centrelines, bridges and cross-sections."""
    height, width = road_visible.shape
    if image is None or image.shape[:2] != (height, width):
        base = np.full((height, width, 3), 35, dtype=np.uint8)
    else:
        base = image.copy()
    colour = base.copy()
    added = road_reconstructed & ~road_visible
    colour[road_visible] = (220, 40, 210)
    colour[added] = (220, 210, 30)
    colour[obstacle] = (20, 40, 230)
    overlay = cv2.addWeighted(base, 0.63, colour, 0.37, 0.0)
    if len(path) >= 2:
        points = np.rint(path).astype(np.int32)
        cv2.polylines(overlay, [points], False, (0, 255, 255), 3, cv2.LINE_AA)
        if cross_section_stride > 0:
            normals = path_normals(path)
            for index in range(0, len(path), cross_section_stride):
                p1 = path[index] + normals[index] * left_width[index]
                p2 = path[index] - normals[index] * right_width[index]
                cv2.line(overlay, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)), (0, 180, 255), 1, cv2.LINE_AA)
    legend = [('visible road GT', (220, 40, 210)), ('reconstructed only', (220, 210, 30)), ('obstacle GT', (20, 40, 230)), ('centreline', (0, 255, 255))]
    y = 34
    for text, bgr in legend:
        cv2.rectangle(overlay, (16, y - 14), (38, y + 8), bgr, -1)
        cv2.putText(overlay, text, (48, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    return overlay

def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    """Save binary mask."""
    unicode_imwrite(path, mask.astype(np.uint8) * 255)

def confidence_label(valid_fraction: float, bridge_count: int, remaining_fragments: int) -> str:
    """Confidence label."""
    if valid_fraction >= 0.65 and bridge_count <= 2 and (remaining_fragments <= 1):
        return 'high'
    if valid_fraction >= 0.35 and remaining_fragments <= 2:
        return 'medium'
    return 'low'

def process_item(item: dict, args: argparse.Namespace) -> dict:
    """Process one annotated image and return its results."""
    key = str(item.get('key') or item.get('file_name') or item.get('name'))
    stem = Path(key).stem
    road_visible = clean_road_mask(rasterize_labels(item, {'road'}), args.min_road_area, args.closing_radius)
    obstacle = rasterize_labels(item, OBSTACLE_LABELS)
    if not road_visible.any():
        raise ValueError('找不到 label=road 的有效 polygon')
    paths = extract_centerline_paths(road_visible, args.min_centerline_length)
    if not paths:
        raise ValueError('road mask 無法抽出足夠長的中心線')
    median_width = estimate_median_width(road_visible, paths)
    joined, bridge_count, gaps = join_centerline_paths(paths, median_width, args.gap_max_widths, args.sample_step)
    centreline = resample_and_smooth(joined, args.sample_step)
    if len(centreline) < 2:
        raise ValueError('中心線長度不足')
    left_width, right_width, valid_width = measure_and_interpolate_widths(road_visible, obstacle, centreline, median_width)
    reconstructed = sweep_road_polygon(road_visible.shape, centreline, left_width, right_width)
    reconstructed |= road_visible
    centreline_mask = np.zeros(road_visible.shape, dtype=np.uint8)
    cv2.polylines(centreline_mask, [np.rint(centreline).astype(np.int32)], False, 255, max(1, int(round(args.sample_step / 2))), cv2.LINE_AA)
    visible_dir = args.output_dir / 'masks' / 'visible_road'
    reconstructed_dir = args.output_dir / 'masks' / 'reconstructed_road'
    centreline_dir = args.output_dir / 'masks' / 'centerline'
    overlay_dir = args.output_dir / 'overlays'
    save_binary_mask(visible_dir / f'{stem}.png', road_visible)
    save_binary_mask(reconstructed_dir / f'{stem}.png', reconstructed)
    unicode_imwrite(centreline_dir / f'{stem}.png', centreline_mask)
    image_path = find_image(args.images_dir, key)
    image = unicode_imread(image_path) if image_path else None
    overlay = make_overlay(image, road_visible, reconstructed, obstacle, centreline, left_width, right_width, args.cross_section_stride)
    unicode_imwrite(overlay_dir / f'{stem}.jpg', overlay)
    visible_area = int(road_visible.sum())
    reconstructed_area = int(reconstructed.sum())
    valid_fraction = float(valid_width.mean()) if len(valid_width) else 0.0
    remaining_fragments = max(1, len(paths) - bridge_count)
    return {'file_name': key, 'road_blockage_gt': item.get('road_blockage', ''), 'status': 'ok', 'road_components': len(paths), 'bridges_added': bridge_count, 'remaining_fragments': remaining_fragments, 'centreline_length_px': round(polyline_length(centreline), 2), 'median_visible_road_width_px': round(median_width, 2), 'mean_gap_length_px': round(float(np.mean(gaps)), 2) if gaps else 0.0, 'visible_road_area_px': visible_area, 'reconstructed_road_area_px': reconstructed_area, 'added_road_area_px': reconstructed_area - visible_area, 'valid_width_fraction': round(valid_fraction, 4), 'reconstruction_confidence': confidence_label(valid_fraction, bridge_count, remaining_fragments), 'image_found': bool(image_path), 'error': ''}
RESULT_FIELDS = ['file_name', 'road_blockage_gt', 'status', 'road_components', 'bridges_added', 'remaining_fragments', 'centreline_length_px', 'median_visible_road_width_px', 'mean_gap_length_px', 'visible_road_area_px', 'reconstructed_road_area_px', 'added_road_area_px', 'valid_width_fraction', 'reconstruction_confidence', 'image_found', 'error']

def error_result(item: dict, message: str) -> dict:
    """Error result."""
    result = {field: '' for field in RESULT_FIELDS}
    result.update({'file_name': item.get('key') or item.get('file_name') or item.get('name', ''), 'road_blockage_gt': item.get('road_blockage', ''), 'status': 'error', 'reconstruction_confidence': 'low', 'error': message})
    return result

def selected_items(items: list[dict], only: Optional[str], maximum: Optional[int]) -> list[dict]:
    """Selected items."""
    if only:
        target = Path(only).stem.lower()
        items = [item for item in items if Path(str(item.get('key') or item.get('file_name') or '')).stem.lower() == target]
        if not items:
            raise ValueError(f'找不到指定影像：{only}')
    return items[:maximum] if maximum is not None else items

def main() -> int:
    """Run the complete dataset inference workflow."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = selected_items(load_items(args.json), args.only, args.max_images)
    results = []
    for index, item in enumerate(items, start=1):
        key = item.get('key') or item.get('file_name') or f'item_{index}'
        print(f'[{index}/{len(items)}] {key}', flush=True)
        try:
            results.append(process_item(item, args))
        except Exception as exc:
            print(f'  ERROR: {exc}', file=sys.stderr, flush=True)
            results.append(error_result(item, str(exc)))
    csv_path = args.output_dir / 'reconstruction_results.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    ok_count = sum((result['status'] == 'ok' for result in results))
    print(f'完成：{ok_count}/{len(results)} 張成功')
    print(f'彙整結果：{csv_path}')
    return 0 if ok_count == len(results) else 1
if __name__ == '__main__':
    raise SystemExit(main())
'''

_ASSESS_CENTERLINE_GRAPH_CORRIDOR_V1_SOURCE = r'''
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
try:
    import cv2
    import numpy as np
    from scipy.ndimage import distance_transform_edt, median_filter
    from skimage.morphology import skeletonize
    import reconstruct_road_v1 as v1
except ImportError as exc:
    raise SystemExit(f'缺少套件或 reconstruct_road_v1.py。請將兩個檔案放在同一個 scripts 資料夾，\n並安裝 numpy、opencv-python、scipy、scikit-image。\n原始錯誤：{exc}') from exc
OBSTACLE_LABELS = {'debris pile', 'fallen vegetation', 'rock'}
CUE_LABELS = ('road line', 'guardrail', 'roadside_wall', 'curb')
LABEL_RANK = {'clear': 0, 'passable_restricted': 1, 'blocked': 2, 'uncertain': -1}

@dataclass
class Fragment:
    fragment_id: int
    path: np.ndarray
    source: str
    median_width: float

@dataclass
class Bridge:
    bridge_id: int
    first_fragment: int
    second_fragment: int
    path: np.ndarray
    gap_px: float
    score: float
    direction_source: str
    maximum_cue_angle_deg: float
    width_ratio: float
    boundary_cross_fraction: float
RESULT_FIELDS = ['file_name', 'road_blockage_gt', 'predicted_label', 'correct', 'status', 'confidence', 'needs_review', 'road_components', 'centreline_fragments', 'accepted_bridges', 'rejected_bridges', 'centreline_coverage', 'width_observation_fraction', 'obstacle_clusters', 'relevant_clusters', 'valid_slices', 'maximum_blockage_p95', 'minimum_free_width_ratio', 'maximum_severe_run_fraction', 'image_found', 'error']
BRIDGE_FIELDS = ['file_name', 'bridge_id', 'first_fragment', 'second_fragment', 'gap_px', 'score', 'direction_source', 'maximum_cue_angle_deg', 'width_ratio', 'boundary_cross_fraction']
SLICE_FIELDS = ['file_name', 'cluster_id', 'route_id', 'path_id', 'slice_index', 'center_x', 'center_y', 'road_width_px', 'obstacle_ratio', 'maximum_free_width_ratio', 'severe']

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description='中心線圖 + 可變路寬道路重建，再以橫切片判定阻塞（V1）')
    p.add_argument('--json', required=True, type=Path)
    p.add_argument('--images-dir', type=Path, default=None)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--only', default=None)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--no-overlays', action='store_true')
    p.add_argument('--save-masks', action='store_true')
    p.add_argument('--min-road-area', type=int, default=300)
    p.add_argument('--closing-radius', type=int, default=2)
    p.add_argument('--min-centreline-length', type=int, default=30)
    p.add_argument('--sample-step-px', type=float, default=5.0)
    p.add_argument('--spur-length-widths', type=float, default=0.55)
    p.add_argument('--maximum-gap-widths', type=float, default=9.0)
    p.add_argument('--maximum-endpoint-angle-deg', type=float, default=52.0)
    p.add_argument('--maximum-roadline-angle-deg', type=float, default=28.0)
    p.add_argument('--maximum-guardrail-angle-deg', type=float, default=35.0)
    p.add_argument('--maximum-wall-curb-angle-deg', type=float, default=45.0)
    p.add_argument('--minimum-width-ratio', type=float, default=0.45)
    p.add_argument('--maximum-width-ratio', type=float, default=2.2)
    p.add_argument('--minimum-curve-radius-widths', type=float, default=1.5)
    p.add_argument('--endpoint-tangent-window-widths', type=float, default=1.0, help='估計端點方向時使用的中心線長度（道路寬倍數）')
    p.add_argument('--maximum-local-turn-deg', type=float, default=32.0, help='平滑曲線內部局部轉角第 95 百分位上限')
    p.add_argument('--curvature-percentile', type=float, default=95.0)
    p.add_argument('--maximum-boundary-cross-fraction', type=float, default=0.035)
    p.add_argument('--cue-search-widths', type=float, default=2.0)
    p.add_argument('--maximum-bridges', type=int, default=12)
    p.add_argument('--roadline-centre-min-depth', type=float, default=0.22, help='road line 到 road 邊界的距離 / 全路寬；達門檻才可當中心位置')
    p.add_argument('--width-ray-widths', type=float, default=1.6)
    p.add_argument('--minimum-side-width-ratio', type=float, default=0.12)
    p.add_argument('--maximum-side-width-ratio', type=float, default=1.25)
    p.add_argument('--guardrail-boundary-blend', type=float, default=0.78)
    p.add_argument('--wall-boundary-blend', type=float, default=0.58)
    p.add_argument('--curb-boundary-blend', type=float, default=0.58)
    p.add_argument('--minimum-obstacle-area', type=int, default=16)
    p.add_argument('--obstacle-cluster-gap-widths', type=float, default=0.2)
    p.add_argument('--road-adjacency-widths', type=float, default=1.25)
    p.add_argument('--assessment-margin-widths', type=float, default=1.25)
    p.add_argument('--slice-step-widths', type=float, default=0.08)
    p.add_argument('--minimum-valid-slices', type=int, default=6)
    p.add_argument('--route-span-widths', type=float, default=0.75, help='障礙互動點前後至少各需延伸的路徑長度')
    p.add_argument('--route-agreement-required', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--clear-maximum-obstacle-ratio', type=float, default=0.3)
    p.add_argument('--severe-obstacle-ratio', type=float, default=0.7)
    p.add_argument('--blocked-severe-slice-fraction', type=float, default=0.3)
    p.add_argument('--blocked-consecutive-fraction', type=float, default=0.2)
    p.add_argument('--blocked-free-width-ratio', type=float, default=0.3)
    p.add_argument('--overlay-max-dimension', type=int, default=1600, help='低記憶體輸出；0 才保留原尺寸')
    p.add_argument('--overlay-jpeg-quality', type=int, default=80)
    p.add_argument('--draw-every-slices', type=int, default=3)
    return p.parse_args()

def normalise(value: np.ndarray) -> Optional[np.ndarray]:
    """Normalise."""
    value = np.asarray(value, dtype=np.float32)
    length = float(np.linalg.norm(value))
    return value / length if length > 1e-06 else None

def acute_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Acute angle deg."""
    first, second = (normalise(first), normalise(second))
    if first is None or second is None:
        return 90.0
    return math.degrees(math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))))

def connected_masks(mask: np.ndarray, minimum_area: int=1) -> list[np.ndarray]:
    """Connected masks."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [labels == i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum_area]

def local_mask_distance(first: np.ndarray, second: np.ndarray, maximum: float) -> float:
    """Local mask distance."""
    yx = np.argwhere(first)
    if not len(yx) or not second.any():
        return float('inf')
    margin = max(3, int(math.ceil(maximum)) + 2)
    y0, y1 = (max(0, int(yx[:, 0].min()) - margin), min(first.shape[0], int(yx[:, 0].max()) + margin + 1))
    x0, x1 = (max(0, int(yx[:, 1].min()) - margin), min(first.shape[1], int(yx[:, 1].max()) + margin + 1))
    a, b = (first[y0:y1, x0:x1], second[y0:y1, x0:x1])
    if not b.any():
        return float('inf')
    distance = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[a].min())

def cluster_obstacles(mask: np.ndarray, road_width: float, args: argparse.Namespace) -> list[np.ndarray]:
    """Cluster obstacles."""
    parts = connected_masks(mask, args.minimum_obstacle_area)
    if not parts:
        return []
    boxes = []
    for part in parts:
        yx = np.argwhere(part)
        boxes.append((int(yx[:, 1].min()), int(yx[:, 0].min()), int(yx[:, 1].max()), int(yx[:, 0].max())))
    parent = list(range(len(parts)))

    def find(i: int) -> int:
        """Find."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        """Union."""
        a, b = (find(a), find(b))
        if a != b:
            parent[b] = a
    limit = 2.0 * max(1.0, args.obstacle_cluster_gap_widths * road_width)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            ax0, ay0, ax1, ay1 = boxes[i]
            bx0, by0, bx1, by1 = boxes[j]
            dx = max(0, ax0 - bx1 - 1, bx0 - ax1 - 1)
            dy = max(0, ay0 - by1 - 1, by0 - ay1 - 1)
            if math.hypot(dx, dy) <= limit:
                union(i, j)
    groups: dict[int, np.ndarray] = {}
    for i, part in enumerate(parts):
        root = find(i)
        groups.setdefault(root, np.zeros_like(mask, dtype=bool))
        groups[root] |= part
    return list(groups.values())

def path_length(path: np.ndarray) -> float:
    """Path length."""
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0

def path_tangent(path: np.ndarray, start: bool, lookahead: int=8) -> np.ndarray:
    """Path tangent."""
    index = min(max(1, int(lookahead)), len(path) - 1)
    value = path[index] - path[0] if start else path[-1] - path[-1 - index]
    result = normalise(value)
    return result if result is not None else np.asarray([1.0, 0.0], dtype=np.float32)

def orient_fragment(fragment: Fragment, endpoint: int, as_first: bool) -> np.ndarray:
    """Orient fragment."""
    if as_first:
        return fragment.path if endpoint == 1 else fragment.path[::-1].copy()
    return fragment.path if endpoint == 0 else fragment.path[::-1].copy()

def resample(path: np.ndarray, step: float) -> np.ndarray:
    """Resample."""
    return v1.resample_and_smooth(path.astype(np.float32), max(2.0, step))

def path_from_mask(mask: np.ndarray, minimum_length: int, step: float) -> list[np.ndarray]:
    """Path from mask."""
    values = v1.extract_centerline_paths(mask, minimum_length)
    return [resample(value, step) for value in values if path_length(value) >= minimum_length]

def centre_roadline_paths(road: np.ndarray, roadline: np.ndarray, road_width: float, args: argparse.Namespace) -> list[np.ndarray]:
    """Centre roadline paths."""
    if not roadline.any():
        return []
    depth = distance_transform_edt(road)
    accepted = np.zeros_like(roadline, dtype=bool)
    for part in connected_masks(roadline, max(8, args.min_centreline_length // 2)):
        on_road = part & road
        overlap = float(on_road.sum()) / max(1, int(part.sum()))
        if overlap < 0.45:
            continue
        values = depth[on_road]
        if len(values) and float(np.median(values)) >= args.roadline_centre_min_depth * road_width:
            accepted |= part
    return path_from_mask(skeletonize(accepted), args.min_centreline_length, args.sample_step_px)

def build_fragments(road: np.ndarray, roadline: np.ndarray, args: argparse.Namespace) -> tuple[list[Fragment], float]:
    """Build visible road and road-line centreline fragments."""
    raw_paths = path_from_mask(road, args.min_centreline_length, args.sample_step_px)
    if not raw_paths:
        return ([], 20.0)
    road_width = max(6.0, v1.estimate_median_width(road, raw_paths))
    minimum_length = max(args.min_centreline_length, args.spur_length_widths * road_width)
    raw_paths = [path for path in raw_paths if path_length(path) >= minimum_length]
    line_paths = centre_roadline_paths(road, roadline, road_width, args)
    road_depth = distance_transform_edt(road)
    fragments: list[Fragment] = []
    for path in raw_paths:
        local_width = local_path_width(road_depth, path, road_width)
        fragments.append(Fragment(len(fragments), path, 'road_mask', local_width))
    for path in line_paths:
        distance = min((polyline_distance(path, value.path) for value in fragments), default=float('inf'))
        if distance > 0.3 * road_width:
            fragments.append(Fragment(len(fragments), path, 'road_line_centre', local_path_width(road_depth, path, road_width)))
    del road_depth
    return (fragments, road_width)

def polyline_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Polyline distance."""
    if not len(first) or not len(second):
        return float('inf')
    a = first[::max(1, len(first) // 100)]
    b = second[::max(1, len(second) // 100)]
    differences = a[:, None, :] - b[None, :, :]
    return float(np.sqrt(np.sum(differences * differences, axis=2)).min())

def local_path_width(distance: np.ndarray, path: np.ndarray, fallback: float) -> float:
    """Local path width."""
    xy = np.rint(path).astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < distance.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < distance.shape[0])
    xy = xy[valid]
    values = 2.0 * distance[xy[:, 1], xy[:, 0]] if len(xy) else np.empty(0)
    values = values[values > 3.0]
    return float(np.median(values)) if len(values) else float(fallback)

def hermite_curve(first: np.ndarray, second: np.ndarray, step: float, tangent_lookahead: int) -> np.ndarray:
    """Hermite curve."""
    p0, p1 = (first[-1], second[0])
    gap = float(np.linalg.norm(p1 - p0))
    count = max(4, int(math.ceil(gap / max(step, 2.0))) + 1)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    scale = min(0.7 * gap, max(0.35 * gap, 3.0 * step))
    m0 = path_tangent(first, False, tangent_lookahead) * scale
    m1 = path_tangent(second, True, tangent_lookahead) * scale
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * p0 + (t ** 3 - 2 * t ** 2 + t) * m0 + (-2 * t ** 3 + 3 * t ** 2) * p1 + (t ** 3 - t ** 2) * m1).astype(np.float32)

def robust_turn_angle(path: np.ndarray, percentile: float) -> float:
    """Robust turn angle."""
    if len(path) < 3:
        return 0.0
    smooth = resample(path, max(2.0, path_length(path) / max(20, len(path) - 1)))
    vectors = np.diff(smooth, axis=0)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-06)
    dots = np.sum(vectors[:-1] * vectors[1:], axis=1)
    turns = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    trim = int(math.floor(0.1 * len(turns)))
    if trim > 0 and len(turns) > 2 * trim:
        turns = turns[trim:-trim]
    return float(np.percentile(turns, np.clip(percentile, 50.0, 100.0))) if len(turns) else 0.0

def sample_path_mask_fraction(path: np.ndarray, mask: np.ndarray) -> float:
    """Sample path mask fraction."""
    xy = np.rint(path).astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    if not valid.any():
        return 1.0
    xy = xy[valid]
    return float(mask[xy[:, 1], xy[:, 0]].mean())

def cue_orientation_near_path(bridge: np.ndarray, cues: dict[str, np.ndarray], road_width: float, args: argparse.Namespace) -> tuple[str, float, float]:
    """Cue orientation near path."""
    tangent = normalise(bridge[-1] - bridge[0])
    if tangent is None:
        return ('none', 90.0, 0.0)
    y0 = max(0, int(np.floor(bridge[:, 1].min() - args.cue_search_widths * road_width)))
    y1 = min(cues['road line'].shape[0], int(np.ceil(bridge[:, 1].max() + args.cue_search_widths * road_width)) + 1)
    x0 = max(0, int(np.floor(bridge[:, 0].min() - args.cue_search_widths * road_width)))
    x1 = min(cues['road line'].shape[1], int(np.ceil(bridge[:, 0].max() + args.cue_search_widths * road_width)) + 1)
    priorities = (('road line', 1.0), ('guardrail', 0.82), ('roadside_wall', 0.62), ('curb', 0.62))
    best = ('none', 90.0, 0.0)
    for name, weight in priorities:
        local = cues[name][y0:y1, x0:x1]
        yx = np.argwhere(local)
        if len(yx) < 10:
            continue
        if len(yx) > 30000:
            yx = yx[::int(math.ceil(len(yx) / 30000))]
        xy = yx[:, ::-1].astype(np.float32)
        covariance = np.cov(xy - xy.mean(axis=0), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        elongation = math.sqrt(float((values[-1] + 1e-06) / (values[0] + 1e-06)))
        if elongation < 1.35:
            continue
        angle = acute_angle_deg(vectors[:, -1], tangent)
        support = weight * min(1.0, len(xy) / max(20.0, road_width)) * min(1.0, elongation / 4.0)
        if support > best[2]:
            best = (name, angle, support)
    return best

def build_bridge_candidates(fragments: list[Fragment], cues: dict[str, np.ndarray], boundary: np.ndarray, road_width: float, args: argparse.Namespace) -> tuple[list[Bridge], list[dict]]:
    """Generate geometrically plausible fragment connections."""
    candidates: list[Bridge] = []
    rejected: list[dict] = []
    maximum_gap = max(30.0, args.maximum_gap_widths * road_width)
    cue_limits = {'road line': args.maximum_roadline_angle_deg, 'guardrail': args.maximum_guardrail_angle_deg, 'roadside_wall': args.maximum_wall_curb_angle_deg, 'curb': args.maximum_wall_curb_angle_deg}
    counter = 0
    tangent_lookahead = max(3, int(round(args.endpoint_tangent_window_widths * road_width / max(args.sample_step_px, 1.0))))
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            width_ratio = max(fragments[i].median_width, fragments[j].median_width) / max(1.0, min(fragments[i].median_width, fragments[j].median_width))
            if min(fragments[i].median_width, fragments[j].median_width) / max(fragments[i].median_width, fragments[j].median_width) < args.minimum_width_ratio or width_ratio > args.maximum_width_ratio:
                continue
            for end_i in (0, 1):
                for end_j in (0, 1):
                    first = orient_fragment(fragments[i], end_i, True)
                    second = orient_fragment(fragments[j], end_j, False)
                    chord = second[0] - first[-1]
                    gap = float(np.linalg.norm(chord))
                    if gap < 2.0 or gap > maximum_gap:
                        continue
                    direction = normalise(chord)
                    if direction is None:
                        continue
                    angle_a = acute_angle_deg(path_tangent(first, False, tangent_lookahead), direction)
                    angle_b = acute_angle_deg(path_tangent(second, True, tangent_lookahead), direction)
                    if max(angle_a, angle_b) > args.maximum_endpoint_angle_deg:
                        rejected.append({'pair': [i, j], 'reason': 'endpoint_angle', 'gap': gap})
                        continue
                    bridge_path = hermite_curve(first, second, args.sample_step_px, tangent_lookahead)
                    cue_source, cue_angle, cue_support = cue_orientation_near_path(bridge_path, cues, road_width, args)
                    if cue_source in cue_limits and cue_support >= 0.18 and (cue_angle > cue_limits[cue_source]):
                        rejected.append({'pair': [i, j], 'reason': f'{cue_source}_angle', 'angle': cue_angle})
                        continue
                    cross = sample_path_mask_fraction(bridge_path, boundary)
                    if cross > args.maximum_boundary_cross_fraction:
                        rejected.append({'pair': [i, j], 'reason': 'boundary_crossing', 'fraction': cross})
                        continue
                    turn = robust_turn_angle(bridge_path, args.curvature_percentile)
                    radius_turn = math.degrees(args.sample_step_px / max(args.minimum_curve_radius_widths * road_width, 1.0)) * 1.75
                    maximum_turn = max(args.maximum_local_turn_deg, radius_turn)
                    if turn > maximum_turn:
                        rejected.append({'pair': [i, j], 'reason': 'curvature', 'turn': turn})
                        continue
                    score = gap / max(road_width, 1.0) + 1.8 * (angle_a + angle_b) / 90.0 + 1.2 * abs(math.log(max(width_ratio, 0.001))) + 5.0 * cross + 0.8 * cue_angle / 90.0 - 1.2 * cue_support
                    counter += 1
                    candidates.append(Bridge(counter, i, j, bridge_path, gap, score, cue_source, cue_angle, width_ratio, cross))
    candidates.sort(key=lambda value: value.score)
    return (candidates, rejected)

def select_bridges(candidates: list[Bridge], fragment_count: int, args: argparse.Namespace) -> list[Bridge]:
    """Select bridges."""
    parent = list(range(fragment_count))

    def find(i: int) -> int:
        """Find."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    selected: list[Bridge] = []
    for value in candidates:
        first, second = (find(value.first_fragment), find(value.second_fragment))
        if first == second:
            continue
        parent[second] = first
        selected.append(value)
        if len(selected) >= args.maximum_bridges:
            break
    return selected

def ray_first(mask: np.ndarray, point: np.ndarray, direction: np.ndarray, maximum: float) -> Optional[float]:
    """Ray first."""
    for distance in np.arange(1.0, maximum + 0.5, 0.75):
        xy = np.rint(point + direction * distance).astype(int)
        if 0 <= xy[0] < mask.shape[1] and 0 <= xy[1] < mask.shape[0] and mask[xy[1], xy[0]]:
            return float(distance)
    return None

def variable_widths(road: np.ndarray, obstacle: np.ndarray, path: np.ndarray, road_width: float, cues: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variable widths."""
    normals = v1.path_normals(path)
    left = np.full(len(path), np.nan, dtype=np.float32)
    right = np.full(len(path), np.nan, dtype=np.float32)
    ray_maximum = max(24.0, args.width_ray_widths * road_width)
    for index, (point, normal) in enumerate(zip(path, normals)):
        if not v1.sample_mask(road, point) or v1.sample_mask(obstacle, point):
            continue
        a = v1.ray_distance(road, point, normal, ray_maximum)
        b = v1.ray_distance(road, point, -normal, ray_maximum)
        total = a + b
        if 0.35 * road_width <= total <= 2.5 * road_width and a > 0 and (b > 0):
            left[index], right[index] = (a, b)
    observed = np.isfinite(left) & np.isfinite(right)

    def fill(values: np.ndarray) -> np.ndarray:
        """Fill."""
        indices = np.arange(len(values), dtype=np.float32)
        valid = np.flatnonzero(np.isfinite(values))
        if not len(valid):
            return np.full(len(values), 0.5 * road_width, dtype=np.float32)
        samples = median_filter(values[valid], size=min(5, len(valid)))
        return np.interp(indices, valid.astype(np.float32), samples).astype(np.float32)
    left, right = (fill(left), fill(right))
    maximum = args.width_ray_widths * road_width
    sources = (('guardrail', args.guardrail_boundary_blend), ('roadside_wall', args.wall_boundary_blend), ('curb', args.curb_boundary_blend))
    for index, (point, normal) in enumerate(zip(path, normals)):
        for sign, target in ((1.0, left), (-1.0, right)):
            best_distance, best_blend = (None, 0.0)
            for name, blend in sources:
                distance = ray_first(cues[name], point, sign * normal, maximum)
                if distance is not None and (best_distance is None or distance < best_distance):
                    best_distance, best_blend = (distance, blend)
            if best_distance is not None and args.minimum_side_width_ratio * road_width <= best_distance <= maximum:
                target[index] = (1.0 - best_blend) * target[index] + best_blend * best_distance
    lower = args.minimum_side_width_ratio * road_width
    upper = args.maximum_side_width_ratio * road_width
    left = np.clip(median_filter(left, size=min(7, max(1, len(left) // 2 * 2 + 1))), lower, upper).astype(np.float32)
    right = np.clip(median_filter(right, size=min(7, max(1, len(right) // 2 * 2 + 1))), lower, upper).astype(np.float32)
    return (left, right, observed)

def reconstruct_corridor(shape: tuple[int, int], road: np.ndarray, obstacle: np.ndarray, fragments: list[Fragment], bridges: list[Bridge], cues: dict[str, np.ndarray], road_width: float, args: argparse.Namespace) -> tuple[np.ndarray, list[dict], float]:
    """Sweep variable road widths along the centreline graph."""
    parent = list(range(len(fragments)))

    def find(index: int) -> int:
        """Find."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for bridge in bridges:
        first, second = (find(bridge.first_fragment), find(bridge.second_fragment))
        if first != second:
            parent[second] = first
    route_ids = {index: find(index) for index in range(len(fragments))}
    paths: list[tuple[str, int, np.ndarray]] = [(f'fragment_{value.fragment_id}', route_ids[value.fragment_id], value.path) for value in fragments]
    paths += [(f'bridge_{value.bridge_id}', route_ids[value.first_fragment], value.path) for value in bridges]
    reconstructed = road.copy()
    geometries: list[dict] = []
    observed_values = []
    for path_id, route_id, path in paths:
        path = resample(path, args.sample_step_px)
        if len(path) < 2:
            continue
        left, right, observed = variable_widths(road, obstacle, path, road_width, cues, args)
        corridor = v1.sweep_road_polygon(shape, path, left, right)
        reconstructed |= corridor
        observed_values.extend(observed.tolist())
        geometries.append({'path_id': path_id, 'route_id': int(route_id), 'path': path, 'left': left, 'right': right})
    fraction = float(np.mean(observed_values)) if observed_values else 0.0
    return (reconstructed, geometries, fraction)

def longest_true_run(values: list[bool]) -> int:
    """Longest true run."""
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best

def longest_false_run_ratio(values: np.ndarray) -> float:
    """Longest false run ratio."""
    best = current = 0
    for value in values.tolist():
        current = 0 if value else current + 1
        best = max(best, current)
    return best / max(1, len(values))

def nearest_path_index_to_mask(path: np.ndarray, mask: np.ndarray, search_margin: float) -> tuple[Optional[int], float]:
    """Nearest path index to mask."""
    yx = np.argwhere(mask)
    if not len(yx) or not len(path):
        return (None, float('inf'))
    margin = max(3, int(math.ceil(search_margin)))
    y0 = max(0, int(yx[:, 0].min()) - margin)
    y1 = min(mask.shape[0], int(yx[:, 0].max()) + margin + 1)
    x0 = max(0, int(yx[:, 1].min()) - margin)
    x1 = min(mask.shape[1], int(yx[:, 1].max()) + margin + 1)
    local = mask[y0:y1, x0:x1]
    distance = cv2.distanceTransform((~local).astype(np.uint8), cv2.DIST_L2, 3)
    xy = np.rint(path).astype(int)
    valid = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return (None, float('inf'))
    local_xy = xy[indices] - np.asarray([x0, y0])
    values = distance[local_xy[:, 1], local_xy[:, 0]]
    selected = int(np.argmin(values))
    return (int(indices[selected]), float(values[selected]))

def route_spans_interaction(route_geometries: list[dict], cluster: np.ndarray, road_width: float, args: argparse.Namespace) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
    """Route spans interaction."""
    nearest = None
    for geometry in route_geometries:
        index, distance = nearest_path_index_to_mask(geometry['path'], cluster, 2.5 * road_width)
        if index is not None and (nearest is None or distance < nearest[0]):
            nearest = (distance, geometry, index)
    if nearest is None:
        return (False, None, None, float('inf'))
    distance, geometry, index = nearest
    path = geometry['path']
    lookahead = max(3, int(round(args.endpoint_tangent_window_widths * road_width / max(args.sample_step_px, 1.0))))
    lower = max(0, index - lookahead)
    upper = min(len(path) - 1, index + lookahead)
    tangent = normalise(path[upper] - path[lower])
    if tangent is None:
        return (False, path[index], None, distance)
    anchor = path[index]
    points = np.vstack([value['path'] for value in route_geometries if len(value['path'])])
    longitudinal = (points - anchor[None, :]) @ tangent
    required = args.route_span_widths * road_width
    spans = float(longitudinal.min(initial=0.0)) <= -required and float(longitudinal.max(initial=0.0)) >= required
    return (spans, anchor, tangent, distance)

def assess_cluster(cluster_id: int, cluster: np.ndarray, geometries: list[dict], road_width: float, args: argparse.Namespace, file_name: str) -> tuple[str, str, str, list[dict], list[dict]]:
    """Assess one obstacle cluster with normal cross-sections."""
    yx = np.argwhere(cluster)
    if not len(yx):
        return ('uncertain', 'low', 'empty_cluster', [], [])
    route_groups: dict[int, list[dict]] = {}
    for geometry in geometries:
        route_groups.setdefault(int(geometry['route_id']), []).append(geometry)
    all_rows: list[dict] = []
    all_drawings: list[dict] = []
    route_results: list[tuple[str, str, str, list[dict], list[dict]]] = []
    step = max(1, int(round(args.slice_step_widths * road_width / max(args.sample_step_px, 1.0))))
    for route_id, route_geometries in route_groups.items():
        spans, anchor, tangent, nearest_distance = route_spans_interaction(route_geometries, cluster, road_width, args)
        if not spans or anchor is None or tangent is None:
            continue
        diagonal = math.hypot(float(np.ptp(yx[:, 1])), float(np.ptp(yx[:, 0])))
        half_length = min(6.0 * road_width, max(2.0 * road_width, 0.5 * diagonal))
        rows: list[dict] = []
        drawings: list[dict] = []
        for geometry in route_geometries:
            path, left, right = (geometry['path'], geometry['left'], geometry['right'])
            normals = v1.path_normals(path)
            longitudinal = (path - anchor[None, :]) @ tangent
            lateral = np.abs((path - anchor[None, :]) @ np.asarray([-tangent[1], tangent[0]], dtype=np.float32))
            indices = np.flatnonzero((np.abs(longitudinal) <= half_length) & (lateral <= 2.5 * road_width))
            for index in indices[::step]:
                width = float(left[index] + right[index])
                sample_count = max(21, int(round(width)) + 1)
                offsets = np.linspace(-float(right[index]), float(left[index]), sample_count, dtype=np.float32)
                points = path[index][None, :] + offsets[:, None] * normals[index][None, :]
                x, y = (np.rint(points[:, 0]).astype(int), np.rint(points[:, 1]).astype(int))
                valid = (x >= 0) & (x < cluster.shape[1]) & (y >= 0) & (y < cluster.shape[0])
                if valid.mean() < 0.9:
                    continue
                occupied = np.zeros(sample_count, dtype=bool)
                occupied[valid] = cluster[y[valid], x[valid]]
                ratio = float(occupied.mean())
                free_ratio = longest_false_run_ratio(occupied)
                severe = ratio >= args.severe_obstacle_ratio or free_ratio <= args.blocked_free_width_ratio
                rows.append({'file_name': file_name, 'cluster_id': cluster_id, 'route_id': route_id, 'path_id': geometry['path_id'], 'slice_index': int(index), 'center_x': round(float(path[index, 0]), 3), 'center_y': round(float(path[index, 1]), 3), 'road_width_px': round(width, 3), 'obstacle_ratio': round(ratio, 6), 'maximum_free_width_ratio': round(free_ratio, 6), 'severe': severe})
                drawings.append({'first': points[0], 'second': points[-1], 'severe': severe, 'ratio': ratio})
        rows.sort(key=lambda row: (row['path_id'], row['slice_index']))
        if len(rows) < args.minimum_valid_slices:
            continue
        ratios = [float(row['obstacle_ratio']) for row in rows]
        severe = [bool(row['severe']) for row in rows]
        severe_fraction = sum(severe) / len(severe)
        path_runs: dict[str, list[bool]] = {}
        for row in rows:
            path_runs.setdefault(str(row['path_id']), []).append(bool(row['severe']))
        longest_run = max((longest_true_run(values) for values in path_runs.values()), default=0)
        run_fraction = longest_run / max(1, len(rows))
        if severe_fraction >= args.blocked_severe_slice_fraction and run_fraction >= args.blocked_consecutive_fraction:
            route_results.append(('blocked', 'medium', 'sustained_severe_cross_sections', rows, drawings))
        elif max(ratios, default=0.0) < args.clear_maximum_obstacle_ratio:
            route_results.append(('clear', 'medium', 'continuous_route_below_clear_threshold', rows, drawings))
        else:
            route_results.append(('passable_restricted', 'medium', 'continuous_route_retains_free_width', rows, drawings))
    if not route_results:
        return ('uncertain', 'low', 'no_continuous_route_across_obstacle', [], [])
    labels = {value[0] for value in route_results}
    if args.route_agreement_required and len(labels) > 1:
        return ('uncertain', 'low', 'continuous_route_disagreement', [], [])
    selected_label = max(labels, key=lambda label: LABEL_RANK[label])
    selected = [value for value in route_results if value[0] == selected_label]
    for value in selected:
        all_rows.extend(value[3])
        all_drawings.extend(value[4])
    confidence = 'high' if len(route_results) > 1 and len(labels) == 1 else 'medium'
    return (selected_label, confidence, selected[0][2], all_rows, all_drawings)

def reduced_imread(path: Path, original_shape: tuple[int, int], maximum: int) -> tuple[Optional[np.ndarray], float]:
    """Reduced imread."""
    longest = max(original_shape)
    factor = 1
    if maximum > 0:
        while factor < 8 and longest / factor > maximum:
            factor *= 2
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[factor]
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)
    except (OSError, cv2.error):
        return (None, 1.0)
    if image is None:
        return (None, 1.0)
    return (image, image.shape[1] / original_shape[1])

def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize mask."""
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

def blend(canvas: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float) -> None:
    """Blend."""
    if not mask.any():
        return
    colour_array = np.asarray(colour, dtype=np.float32)
    canvas[mask] = np.clip((1.0 - alpha) * canvas[mask] + alpha * colour_array, 0, 255).astype(np.uint8)

def save_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    """Save jpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 30, 100))])
    if not ok:
        raise OSError(f'無法編碼：{path}')
    data.tofile(str(path))

def make_overlay(image_path: Optional[Path], road: np.ndarray, obstacles: np.ndarray, reconstructed: np.ndarray, geometries: list[dict], bridges: list[Bridge], drawings: list[dict], predicted: str, confidence: str, args: argparse.Namespace, draw_header: bool=True) -> np.ndarray:
    """Draw masks, centrelines, bridges and cross-sections."""
    image, scale = reduced_imread(image_path, road.shape, args.overlay_max_dimension) if image_path else (None, 1.0)
    if image is None:
        scale = min(1.0, args.overlay_max_dimension / max(road.shape)) if args.overlay_max_dimension > 0 else 1.0
        image = np.full((max(1, int(round(road.shape[0] * scale))), max(1, int(round(road.shape[1] * scale))), 3), 45, np.uint8)
    shape = image.shape[:2]
    blend(image, resize_mask(road, shape), (70, 210, 90), 0.25)
    blend(image, resize_mask(reconstructed & ~road, shape), (255, 210, 40), 0.32)
    blend(image, resize_mask(obstacles, shape), (45, 45, 235), 0.42)
    thickness = max(1, int(round(max(shape) / 1200)))
    for geometry in geometries:
        points = np.rint(geometry['path'] * scale).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points], False, (0, 255, 255), 2 * thickness, cv2.LINE_AA)
    for bridge in bridges:
        points = np.rint(bridge.path * scale).astype(np.int32)
        cv2.polylines(image, [points], False, (255, 0, 255), 3 * thickness, cv2.LINE_AA)
    for drawing in drawings[::max(1, args.draw_every_slices)]:
        colour = (0, 0, 255) if drawing['severe'] else (255, 255, 255)
        first = tuple(np.rint(drawing['first'] * scale).astype(int))
        second = tuple(np.rint(drawing['second'] * scale).astype(int))
        cv2.line(image, first, second, colour, thickness, cv2.LINE_AA)
    if draw_header:
        cv2.rectangle(image, (0, 0), (min(shape[1] - 1, 900), min(shape[0] - 1, 72)), (0, 0, 0), -1)
        colour = {'clear': (60, 230, 80), 'passable_restricted': (0, 210, 255), 'blocked': (40, 40, 255), 'uncertain': (255, 90, 255)}[predicted]
        cv2.putText(image, f'centreline graph V1.1: {predicted}', (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, colour, 2, cv2.LINE_AA)
        cv2.putText(image, f'confidence={confidence} bridges={len(bridges)}', (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    return image

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path: Path) -> list[dict]:
    """Read csv."""
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))

def process_item(item: dict, args: argparse.Namespace) -> tuple[dict, list[dict], list[dict]]:
    """Process one annotated image and return its results."""
    key = str(item.get('key') or item.get('file_name') or item.get('name'))
    stem = Path(key).stem
    road = v1.clean_road_mask(v1.rasterize_labels(item, {'road'}), args.min_road_area, args.closing_radius)
    if not road.any():
        raise ValueError('找不到有效 road polygon')
    obstacles = v1.rasterize_labels(item, OBSTACLE_LABELS)
    cues = {name: v1.rasterize_labels(item, {name}) for name in CUE_LABELS}
    fragments, road_width = build_fragments(road, cues['road line'], args)
    if not fragments:
        raise ValueError('找不到有效道路中心線')
    boundary = cues['guardrail'] | cues['roadside_wall'] | cues['curb']
    candidates, rejected = build_bridge_candidates(fragments, cues, boundary, road_width, args)
    bridges = select_bridges(candidates, len(fragments), args)
    reconstructed, geometries, observed_fraction = reconstruct_corridor(road.shape, road, obstacles, fragments, bridges, cues, road_width, args)
    clusters = cluster_obstacles(obstacles, road_width, args)
    adjacency = args.road_adjacency_widths * road_width
    relevant = [value for value in clusters if local_mask_distance(value, reconstructed, adjacency) <= adjacency]
    slice_rows: list[dict] = []
    drawings: list[dict] = []
    cluster_results = []
    for cluster_id, cluster in enumerate(relevant, start=1):
        label, confidence, reason, rows, local_drawings = assess_cluster(cluster_id, cluster, geometries, road_width, args, key)
        slice_rows.extend(rows)
        drawings.extend(local_drawings)
        cluster_results.append((label, confidence, reason, rows))
    if not clusters:
        predicted, confidence, reason = ('clear', 'high', 'no_obstacle_cluster')
    elif not relevant:
        predicted, confidence, reason = ('uncertain', 'low', 'obstacle_without_reconstructed_road_interaction')
    else:
        certain = [value for value in cluster_results if value[0] != 'uncertain']
        if not certain:
            predicted, confidence, reason = ('uncertain', 'low', 'all_clusters_lack_valid_cross_sections')
        else:
            worst = max(certain, key=lambda value: LABEL_RANK[value[0]])
            unresolved = any((value[0] == 'uncertain' for value in cluster_results))
            if unresolved and worst[0] != 'blocked':
                predicted, confidence, reason = ('uncertain', 'low', 'unresolved_safety_relevant_cluster')
            else:
                predicted, confidence, reason = (worst[0], worst[1], worst[2])
    image_path = v1.find_image(args.images_dir, key)
    if args.save_masks:
        v1.unicode_imwrite(args.output_dir / 'masks' / f'{stem}.png', reconstructed.astype(np.uint8) * 255)
    if not args.no_overlays:
        overlay = make_overlay(image_path, road, obstacles, reconstructed, geometries, bridges, drawings, predicted, confidence, args)
        save_jpeg(args.output_dir / 'overlays' / f'{stem}.jpg', overlay, args.overlay_jpeg_quality)
        del overlay
    diagnostic_path = args.output_dir / 'diagnostics' / f'{stem}.json'
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic = {'file_name': key, 'prediction': predicted, 'confidence': confidence, 'reason': reason, 'global_road_width_px': road_width, 'method': 'direction_constrained_centreline_graph_variable_width_corridor_v1_1', 'endpoint_pairing_used': False, 'fragments': [{'id': value.fragment_id, 'source': value.source, 'length_px': path_length(value.path), 'median_width_px': value.median_width} for value in fragments], 'accepted_bridges': [{**asdict(value), 'path': None} for value in bridges], 'rejected_candidates': rejected[:500], 'cluster_results': [{'label': value[0], 'confidence': value[1], 'reason': value[2], 'valid_slices': len(value[3])} for value in cluster_results]}
    with diagnostic_path.open('w', encoding='utf-8') as file:
        json.dump(diagnostic, file, ensure_ascii=False, indent=2)
    road_skeleton_length = sum((path_length(value.path) for value in fragments))
    bridge_length = sum((path_length(value.path) for value in bridges))
    coverage = road_skeleton_length / max(1.0, road_skeleton_length + bridge_length)
    ratios = [float(row['obstacle_ratio']) for row in slice_rows]
    free = [float(row['maximum_free_width_ratio']) for row in slice_rows]
    severe = [bool(row['severe']) for row in slice_rows]
    gt = str(item.get('road_blockage', ''))
    result = {'file_name': key, 'road_blockage_gt': gt, 'predicted_label': predicted, 'correct': predicted == gt, 'status': 'needs_review' if predicted == 'uncertain' else 'ok', 'confidence': confidence, 'needs_review': predicted == 'uncertain', 'road_components': len(connected_masks(road, args.min_road_area)), 'centreline_fragments': len(fragments), 'accepted_bridges': len(bridges), 'rejected_bridges': len(rejected), 'centreline_coverage': round(coverage, 6), 'width_observation_fraction': round(observed_fraction, 6), 'obstacle_clusters': len(clusters), 'relevant_clusters': len(relevant), 'valid_slices': len(slice_rows), 'maximum_blockage_p95': round(float(np.percentile(ratios, 95)), 6) if ratios else 0.0, 'minimum_free_width_ratio': round(min(free), 6) if free else 1.0, 'maximum_severe_run_fraction': round(longest_true_run(severe) / max(1, len(severe)), 6), 'image_found': image_path is not None, 'error': ''}
    bridge_rows = [{'file_name': key, **{name: value for name, value in asdict(bridge).items() if name != 'path'}} for bridge in bridges]
    return (result, bridge_rows, slice_rows)

def error_result(item: dict, message: str) -> dict:
    """Error result."""
    result = {field: '' for field in RESULT_FIELDS}
    result.update({'file_name': item.get('key') or item.get('file_name') or item.get('name', ''), 'road_blockage_gt': item.get('road_blockage', ''), 'predicted_label': 'uncertain', 'correct': False, 'status': 'error', 'confidence': 'low', 'needs_review': True, 'error': message})
    return result

def validate_args(args: argparse.Namespace) -> None:
    """Validate args."""
    ratios = [args.minimum_width_ratio, args.roadline_centre_min_depth, args.minimum_side_width_ratio, args.maximum_side_width_ratio, args.clear_maximum_obstacle_ratio, args.severe_obstacle_ratio, args.blocked_severe_slice_fraction, args.blocked_consecutive_fraction, args.blocked_free_width_ratio]
    if any((value < 0.0 for value in ratios)):
        raise ValueError('比例參數不可小於 0')
    if args.clear_maximum_obstacle_ratio >= args.severe_obstacle_ratio:
        raise ValueError('clear 門檻必須小於 severe 門檻')
    if args.overlay_max_dimension and args.overlay_max_dimension < 320:
        raise ValueError('--overlay-max-dimension 建議至少為 320')

def main() -> int:
    """Run the complete dataset inference workflow."""
    args = parse_args()
    validate_args(args)
    items = v1.load_items(args.json)
    if args.only:
        target = Path(args.only).stem.lower()
        items = [item for item in items if Path(str(item.get('key') or item.get('file_name') or item.get('name', ''))).stem.lower() == target]
        if not items:
            raise ValueError(f'找不到指定影像：{args.only}')
    if args.max_images is not None:
        items = items[:args.max_images]
    result_path = args.output_dir / 'centreline_graph_corridor_v1_1_results.csv'
    bridge_path = args.output_dir / 'bridge_assessments.csv'
    slice_path = args.output_dir / 'slice_profiles.csv'
    results = read_csv(result_path) if args.resume else []
    bridge_rows = read_csv(bridge_path) if args.resume else []
    slice_rows = read_csv(slice_path) if args.resume else []
    if args.resume:
        statuses = {str(row.get('file_name', '')): str(row.get('status', '')) for row in results}
        completed = {name for name, status in statuses.items() if not (args.retry_errors and status == 'error')}
        pending = [item for item in items if str(item.get('key') or item.get('file_name') or item.get('name', '')) not in completed]
        pending_names = {str(item.get('key') or item.get('file_name') or item.get('name', '')) for item in pending}
        results = [row for row in results if str(row.get('file_name', '')) not in pending_names]
        bridge_rows = [row for row in bridge_rows if str(row.get('file_name', '')) not in pending_names]
        slice_rows = [row for row in slice_rows if str(row.get('file_name', '')) not in pending_names]
        print(f'resume: 已保留 {len(completed)} 張，待處理 {len(pending)} 張', flush=True)
        items = pending
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = len(items)
    for index, item in enumerate(items, start=1):
        key = str(item.get('key') or item.get('file_name') or item.get('name', f'item_{index}'))
        print(f'[{index}/{total}] {key}', flush=True)
        try:
            result, bridges, slices = process_item(item, args)
            print(f"  {result['predicted_label']} bridges={result['accepted_bridges']} slices={result['valid_slices']}", flush=True)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__
            print(f'  ERROR: {message}', flush=True)
            result, bridges, slices = (error_result(item, message), [], [])
        results.append(result)
        bridge_rows.extend(bridges)
        slice_rows.extend(slices)
        write_csv(result_path, results, RESULT_FIELDS)
        write_csv(bridge_path, bridge_rows, BRIDGE_FIELDS)
        write_csv(slice_path, slice_rows, SLICE_FIELDS)
        gc.collect()
    valid = [row for row in results if row.get('status') != 'error']
    accuracy = sum((str(row.get('correct', '')).lower() == 'true' for row in valid)) / max(1, len(valid))
    print(f'完成：valid={len(valid)} error={len(results) - len(valid)} accuracy={accuracy:.4f}')
    print(f'結果：{result_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
'''

_ASSESS_CENTRELINE_GRAPH_CORRIDOR_V1_2_MAIN_TRUNK_SOURCE = r'''
from __future__ import annotations
import csv
import gc
import heapq
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional
try:
    import cv2
    import numpy as np
    import reconstruct_road_v1 as v1
    import assess_centerline_graph_corridor_v1 as c1
except ImportError as exc:
    raise SystemExit(f'缺少套件或 companion script。請將本程式、reconstruct_road_v1.py、\nassess_centerline_graph_corridor_v1.py 放在同一個 scripts/。\n原始錯誤：{exc}') from exc
METHOD = 'centreline_graph_corridor_v1_2_unlimited_gap_single_main_trunk'
RESULT_FIELDS = [*c1.RESULT_FIELDS[:-1], 'all_fragments', 'main_trunk_fragments', 'all_connection_candidates', 'main_trunk_edges', 'terminal_source', 'main_trunk_cost', 'maximum_selected_gap_px', 'maximum_selected_gap_widths', 'selected_obstacle_support', 'selected_boundary_cross_fraction', 'error']
CANDIDATE_FIELDS = ['file_name', 'candidate_id', 'first_fragment', 'first_endpoint', 'second_fragment', 'second_endpoint', 'gap_px', 'gap_widths', 'endpoint_angle_a_deg', 'endpoint_angle_b_deg', 'width_ratio', 'curvature_deg', 'cue_source', 'cue_angle_deg', 'cue_support', 'obstacle_support', 'boundary_cross_fraction', 'cost', 'selected']

def parse_args():
    """Parse command-line arguments."""
    args = c1.parse_args()
    args.main_trunk_min_length_widths = 0.75
    args.significant_fragment_length_ratio = 0.2
    args.terminal_border_widths = 1.25
    args.candidates_per_fragment_pair = 4
    args.distance_cost_weight = 1.0
    args.endpoint_angle_cost_weight = 2.2
    args.width_cost_weight = 1.4
    args.curvature_cost_weight = 1.1
    args.boundary_cost_weight = 22.0
    args.boundary_violation_cost = 18.0
    args.cue_angle_cost_weight = 1.0
    args.cue_support_reward = 1.8
    args.obstacle_support_reward = 2.6
    args.no_occlusion_long_gap_cost = 2.5
    return args

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path: Path) -> list[dict]:
    """Read csv."""
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))

def build_unlimited_candidates(fragments: list[c1.Fragment], cues: dict[str, np.ndarray], boundary: np.ndarray, obstacles: np.ndarray, road_width: float, args) -> tuple[list[c1.Bridge], list[dict], dict[int, tuple[int, int]]]:
    """Generate main-trunk connections without a fixed gap limit."""
    candidates: list[c1.Bridge] = []
    rows: list[dict] = []
    endpoint_map: dict[int, tuple[int, int]] = {}
    tangent_lookahead = max(3, int(round(args.endpoint_tangent_window_widths * road_width / max(args.sample_step_px, 1.0))))
    counter = 0
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            width_ratio = max(fragments[i].median_width, fragments[j].median_width) / max(1.0, min(fragments[i].median_width, fragments[j].median_width))
            pair_values = []
            for end_i in (0, 1):
                for end_j in (0, 1):
                    first = c1.orient_fragment(fragments[i], end_i, True)
                    second = c1.orient_fragment(fragments[j], end_j, False)
                    chord = second[0] - first[-1]
                    gap = float(np.linalg.norm(chord))
                    if gap < 2.0:
                        continue
                    direction = c1.normalise(chord)
                    if direction is None:
                        continue
                    angle_a = c1.acute_angle_deg(c1.path_tangent(first, False, tangent_lookahead), direction)
                    angle_b = c1.acute_angle_deg(c1.path_tangent(second, True, tangent_lookahead), direction)
                    bridge_path = c1.hermite_curve(first, second, args.sample_step_px, tangent_lookahead)
                    curvature = c1.robust_turn_angle(bridge_path, args.curvature_percentile)
                    cross = c1.sample_path_mask_fraction(bridge_path, boundary)
                    obstacle_support = c1.sample_path_mask_fraction(bridge_path, obstacles)
                    geometric_cost = args.distance_cost_weight * gap / max(road_width, 1.0) + args.endpoint_angle_cost_weight * (angle_a + angle_b) / 180.0 + args.width_cost_weight * abs(math.log(max(width_ratio, 0.001))) + args.curvature_cost_weight * curvature / 45.0 + args.boundary_cost_weight * cross + (args.boundary_violation_cost if cross > args.maximum_boundary_cross_fraction else 0.0) - args.obstacle_support_reward * min(0.6, obstacle_support)
                    if gap > args.maximum_gap_widths * road_width and obstacle_support < 0.03:
                        geometric_cost += args.no_occlusion_long_gap_cost
                    pair_values.append({'end_i': end_i, 'end_j': end_j, 'gap': gap, 'path': bridge_path, 'angle_a': angle_a, 'angle_b': angle_b, 'curvature': curvature, 'cross': cross, 'obstacle_support': obstacle_support, 'width_ratio': width_ratio, 'geometric_cost': geometric_cost})
            pair_values.sort(key=lambda value: value['geometric_cost'])
            for value in pair_values[:args.candidates_per_fragment_pair]:
                cue_source, cue_angle, cue_support = c1.cue_orientation_near_path(value['path'], cues, road_width, args)
                cost = value['geometric_cost'] + args.cue_angle_cost_weight * cue_angle / 90.0 - args.cue_support_reward * cue_support
                cost = max(0.05, float(cost))
                counter += 1
                bridge = c1.Bridge(counter, i, j, value['path'].astype(np.float32), value['gap'], cost, cue_source, cue_angle, width_ratio, value['cross'])
                candidates.append(bridge)
                endpoint_map[counter] = (value['end_i'], value['end_j'])
                rows.append({'candidate_id': counter, 'first_fragment': i, 'first_endpoint': value['end_i'], 'second_fragment': j, 'second_endpoint': value['end_j'], 'gap_px': round(value['gap'], 4), 'gap_widths': round(value['gap'] / max(road_width, 1.0), 6), 'endpoint_angle_a_deg': round(value['angle_a'], 4), 'endpoint_angle_b_deg': round(value['angle_b'], 4), 'width_ratio': round(width_ratio, 6), 'curvature_deg': round(value['curvature'], 4), 'cue_source': cue_source, 'cue_angle_deg': round(cue_angle, 4), 'cue_support': round(cue_support, 6), 'obstacle_support': round(value['obstacle_support'], 6), 'boundary_cross_fraction': round(value['cross'], 6), 'cost': round(cost, 6), 'selected': False})
    candidates.sort(key=lambda value: value.score)
    return (candidates, rows, endpoint_map)

def endpoint_point(fragment: c1.Fragment, endpoint: int) -> np.ndarray:
    """Endpoint point."""
    return fragment.path[0] if endpoint == 0 else fragment.path[-1]

def endpoint_tangent_outward(fragment: c1.Fragment, endpoint: int, lookahead: int) -> np.ndarray:
    """Endpoint tangent outward."""
    path = fragment.path
    index = min(max(1, lookahead), len(path) - 1)
    vector = path[0] - path[index] if endpoint == 0 else path[-1] - path[-1 - index]
    value = c1.normalise(vector)
    return value if value is not None else np.asarray([1.0, 0.0], np.float32)

def choose_terminals(fragments: list[c1.Fragment], shape: tuple[int, int], road_width: float, args) -> tuple[tuple[int, int], tuple[int, int], str]:
    """Choose terminals."""
    if len(fragments) == 1:
        return ((0, 0), (0, 1), 'single_fragment')
    lengths = np.asarray([c1.path_length(value.path) for value in fragments])
    maximum = float(lengths.max(initial=0.0))
    significant = [index for index, length in enumerate(lengths) if length >= max(args.main_trunk_min_length_widths * road_width, args.significant_fragment_length_ratio * maximum)]
    if len(significant) < 2:
        significant = list(np.argsort(lengths)[-min(2, len(fragments)):])
    height, width = shape
    border_limit = args.terminal_border_widths * road_width
    endpoints = []
    for fragment_id in significant:
        fragment = fragments[fragment_id]
        for endpoint in (0, 1):
            point = endpoint_point(fragment, endpoint)
            border_distance = min(float(point[0]), float(point[1]), float(width - 1 - point[0]), float(height - 1 - point[1]))
            endpoints.append((fragment_id, endpoint, point, border_distance))
    border = [value for value in endpoints if value[3] <= border_limit]
    pool = border if len({value[0] for value in border}) >= 2 else endpoints
    source = 'image_border' if pool is border else 'farthest_significant_fragments'
    lookahead = max(3, int(round(args.endpoint_tangent_window_widths * road_width / max(args.sample_step_px, 1.0))))
    best = None
    diagonal = math.hypot(width, height)
    for index, first in enumerate(pool):
        for second in pool[index + 1:]:
            if first[0] == second[0] and len(significant) >= 2:
                continue
            distance = float(np.linalg.norm(second[2] - first[2]))
            direction = c1.normalise(second[2] - first[2])
            if direction is None:
                continue
            tangent_a = endpoint_tangent_outward(fragments[first[0]], first[1], lookahead)
            tangent_b = endpoint_tangent_outward(fragments[second[0]], second[1], lookahead)
            alignment = (c1.acute_angle_deg(tangent_a, direction) + c1.acute_angle_deg(tangent_b, direction)) / 180.0
            width_ratio = max(fragments[first[0]].median_width, fragments[second[0]].median_width) / max(1.0, min(fragments[first[0]].median_width, fragments[second[0]].median_width))
            length_support = (lengths[first[0]] + lengths[second[0]]) / max(1.0, lengths.sum())
            score = distance / max(diagonal, 1.0) + 0.3 * length_support - 0.2 * alignment - 0.1 * abs(math.log(max(width_ratio, 0.001)))
            if best is None or score > best[0]:
                best = (score, (first[0], first[1]), (second[0], second[1]))
    if best is None:
        longest = int(np.argmax(lengths))
        return ((longest, 0), (longest, 1), 'longest_fragment_fallback')
    return (best[1], best[2], source)

def search_main_trunk(fragments: list[c1.Fragment], candidates: list[c1.Bridge], endpoint_map: dict[int, tuple[int, int]], shape: tuple[int, int], road_width: float, args) -> tuple[list[int], list[c1.Bridge], str, float, tuple, tuple]:
    """Search the fragment graph for one primary road trunk."""
    start, goal, terminal_source = choose_terminals(fragments, shape, road_width, args)
    if start == goal:
        return ([start[0]], [], terminal_source, 0.0, start, goal)
    graph: dict[tuple[int, int], list[tuple[float, tuple[int, int], Optional[c1.Bridge]]]] = {}

    def add(first, second, cost, bridge):
        """Add."""
        graph.setdefault(first, []).append((float(cost), second, bridge))
        graph.setdefault(second, []).append((float(cost), first, bridge))
    for fragment in fragments:
        add((fragment.fragment_id, 0), (fragment.fragment_id, 1), 0.02, None)
    for bridge in candidates:
        end_i, end_j = endpoint_map[bridge.bridge_id]
        add((bridge.first_fragment, end_i), (bridge.second_fragment, end_j), bridge.score, bridge)
    distances = {start: 0.0}
    previous: dict[tuple[int, int], tuple[tuple[int, int], Optional[c1.Bridge]]] = {}
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distances.get(node, float('inf')) + 1e-09:
            continue
        if node == goal:
            break
        for edge_cost, neighbour, bridge in graph.get(node, []):
            new_cost = cost + edge_cost
            if new_cost + 1e-09 >= distances.get(neighbour, float('inf')):
                continue
            distances[neighbour] = new_cost
            previous[neighbour] = (node, bridge)
            heapq.heappush(queue, (new_cost, neighbour))
    if goal not in distances:
        longest = max(fragments, key=lambda value: c1.path_length(value.path))
        return ([longest.fragment_id], [], 'disconnected_graph_fallback', float('inf'), start, goal)
    nodes, bridges = ([goal], [])
    current = goal
    while current != start:
        parent, bridge = previous[current]
        if bridge is not None:
            bridges.append(bridge)
        nodes.append(parent)
        current = parent
    nodes.reverse()
    bridges.reverse()
    fragment_ids = sorted({node[0] for node in nodes})
    unique = []
    seen = set()
    for bridge in bridges:
        if bridge.bridge_id not in seen:
            unique.append(bridge)
            seen.add(bridge.bridge_id)
    return (fragment_ids, unique, terminal_source, float(distances[goal]), start, goal)

def remap_trunk(fragments: list[c1.Fragment], selected_ids: list[int], bridges: list[c1.Bridge]) -> tuple[list[c1.Fragment], list[c1.Bridge]]:
    """Remap the selected trunk fragments and bridges."""
    mapping = {old: new for new, old in enumerate(selected_ids)}
    selected_fragments = [c1.Fragment(mapping[old], fragments[old].path.copy(), fragments[old].source, fragments[old].median_width) for old in selected_ids]
    selected_bridges = []
    for bridge in bridges:
        if bridge.first_fragment not in mapping or bridge.second_fragment not in mapping:
            continue
        selected_bridges.append(replace(bridge, first_fragment=mapping[bridge.first_fragment], second_fragment=mapping[bridge.second_fragment]))
    return (selected_fragments, selected_bridges)

def make_overlay(image_path: Optional[Path], road: np.ndarray, obstacles: np.ndarray, reconstructed: np.ndarray, geometries: list[dict], bridges: list[c1.Bridge], drawings: list[dict], predicted: str, confidence: str, terminal_source: str, trunk_cost: float, args) -> np.ndarray:
    """Draw masks, centrelines, bridges and cross-sections."""
    overlay = c1.make_overlay(image_path, road, obstacles, reconstructed, geometries, bridges, drawings, predicted, confidence, args)
    cv2.rectangle(overlay, (0, 0), (min(overlay.shape[1] - 1, 1050), min(overlay.shape[0] - 1, 76)), (0, 0, 0), -1)
    colour = {'clear': (60, 230, 80), 'passable_restricted': (0, 210, 255), 'blocked': (40, 40, 255), 'uncertain': (255, 90, 255)}[predicted]
    cv2.putText(overlay, f'centreline graph V1.2 main trunk: {predicted}', (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.64, colour, 2, cv2.LINE_AA)
    cost_text = 'inf' if not np.isfinite(trunk_cost) else f'{trunk_cost:.2f}'
    cv2.putText(overlay, f'confidence={confidence} source={terminal_source} bridges={len(bridges)} cost={cost_text}', (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (230, 230, 230), 1, cv2.LINE_AA)
    return overlay

def process_item(item: dict, args):
    """Process one annotated image and return its results."""
    key = str(item.get('key') or item.get('file_name') or item.get('name'))
    stem = Path(key).stem
    road = v1.clean_road_mask(v1.rasterize_labels(item, {'road'}), args.min_road_area, args.closing_radius)
    if not road.any():
        raise ValueError('找不到有效 road polygon')
    obstacles = v1.rasterize_labels(item, c1.OBSTACLE_LABELS)
    cues = {name: v1.rasterize_labels(item, {name}) for name in c1.CUE_LABELS}
    all_fragments, road_width = c1.build_fragments(road, cues['road line'], args)
    if not all_fragments:
        raise ValueError('找不到有效道路中心線')
    boundary = cues['guardrail'] | cues['roadside_wall'] | cues['curb']
    candidates, candidate_rows, endpoint_map = build_unlimited_candidates(all_fragments, cues, boundary, obstacles, road_width, args)
    selected_ids, selected_bridges_old, terminal_source, trunk_cost, start, goal = search_main_trunk(all_fragments, candidates, endpoint_map, road.shape, road_width, args)
    fragments, bridges = remap_trunk(all_fragments, selected_ids, selected_bridges_old)
    selected_candidate_ids = {value.bridge_id for value in selected_bridges_old}
    for row in candidate_rows:
        row['file_name'] = key
        row['selected'] = int(row['candidate_id']) in selected_candidate_ids
    selected_rows = [row for row in candidate_rows if row['selected']]
    weak_long_connection = any((float(row['gap_widths']) > args.maximum_gap_widths and float(row['obstacle_support']) < 0.03 and (float(row['cue_support']) < 0.18) for row in selected_rows))
    boundary_violation = any((float(row['boundary_cross_fraction']) > args.maximum_boundary_cross_fraction for row in selected_rows))
    geometry_review = weak_long_connection or boundary_violation
    reconstructed, geometries, observed_fraction = c1.reconstruct_corridor(road.shape, road, obstacles, fragments, bridges, cues, road_width, args)
    clusters = c1.cluster_obstacles(obstacles, road_width, args)
    adjacency = args.road_adjacency_widths * road_width
    relevant = [value for value in clusters if c1.local_mask_distance(value, reconstructed, adjacency) <= adjacency]
    slice_rows, drawings, cluster_results = ([], [], [])
    for cluster_id, cluster in enumerate(relevant, start=1):
        label, confidence, reason, rows, local_drawings = c1.assess_cluster(cluster_id, cluster, geometries, road_width, args, key)
        slice_rows.extend(rows)
        drawings.extend(local_drawings)
        cluster_results.append((label, confidence, reason, rows))
    if not clusters:
        predicted, confidence, reason = ('clear', 'high', 'no_obstacle_cluster')
    elif not relevant:
        predicted, confidence, reason = ('uncertain', 'low', 'obstacle_without_main_trunk_interaction')
    else:
        certain = [value for value in cluster_results if value[0] != 'uncertain']
        if not certain:
            predicted, confidence, reason = ('uncertain', 'low', 'main_trunk_lacks_valid_cross_sections')
        else:
            worst = max(certain, key=lambda value: c1.LABEL_RANK[value[0]])
            unresolved = any((value[0] == 'uncertain' for value in cluster_results))
            if unresolved and worst[0] != 'blocked':
                predicted, confidence, reason = ('uncertain', 'low', 'unresolved_main_trunk_cluster')
            else:
                predicted, confidence, reason = (worst[0], worst[1], worst[2])
    if predicted != 'uncertain' and geometry_review:
        confidence = 'low'
        reason += ';forced_main_trunk_requires_review'
    image_path = v1.find_image(args.images_dir, key)
    if args.save_masks:
        v1.unicode_imwrite(args.output_dir / 'masks' / f'{stem}.png', reconstructed.astype(np.uint8) * 255)
    if not args.no_overlays:
        overlay = make_overlay(image_path, road, obstacles, reconstructed, geometries, bridges, drawings, predicted, confidence, terminal_source, trunk_cost, args)
        c1.save_jpeg(args.output_dir / 'overlays' / f'{stem}.jpg', overlay, args.overlay_jpeg_quality)
        del overlay
    diagnostic_path = args.output_dir / 'diagnostics' / f'{stem}.json'
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostic_path.open('w', encoding='utf-8') as file:
        json.dump({'file_name': key, 'prediction': predicted, 'confidence': confidence, 'reason': reason, 'method': METHOD, 'road_width_px': road_width, 'terminal_source': terminal_source, 'terminal_start': list(start), 'terminal_goal': list(goal), 'main_trunk_cost': trunk_cost if np.isfinite(trunk_cost) else None, 'all_fragments': [{'id': value.fragment_id, 'source': value.source, 'length_px': c1.path_length(value.path), 'median_width_px': value.median_width} for value in all_fragments], 'selected_fragment_ids': selected_ids, 'selected_connections': selected_rows, 'cluster_results': [{'label': value[0], 'confidence': value[1], 'reason': value[2], 'valid_slices': len(value[3])} for value in cluster_results]}, file, ensure_ascii=False, indent=2)
    road_length = sum((c1.path_length(value.path) for value in fragments))
    bridge_length = sum((c1.path_length(value.path) for value in bridges))
    ratios = [float(row['obstacle_ratio']) for row in slice_rows]
    free = [float(row['maximum_free_width_ratio']) for row in slice_rows]
    severe = [bool(row['severe']) for row in slice_rows]
    gt = str(item.get('road_blockage', ''))
    maximum_gap = max((value.gap_px for value in bridges), default=0.0)
    result = {'file_name': key, 'road_blockage_gt': gt, 'predicted_label': predicted, 'correct': predicted == gt, 'status': 'needs_review' if predicted == 'uncertain' or geometry_review else 'ok', 'confidence': confidence, 'needs_review': predicted == 'uncertain' or geometry_review, 'road_components': len(c1.connected_masks(road, args.min_road_area)), 'centreline_fragments': len(fragments), 'accepted_bridges': len(bridges), 'rejected_bridges': 0, 'centreline_coverage': round(road_length / max(1.0, road_length + bridge_length), 6), 'width_observation_fraction': round(observed_fraction, 6), 'obstacle_clusters': len(clusters), 'relevant_clusters': len(relevant), 'valid_slices': len(slice_rows), 'maximum_blockage_p95': round(float(np.percentile(ratios, 95)), 6) if ratios else 0.0, 'minimum_free_width_ratio': round(min(free), 6) if free else 1.0, 'maximum_severe_run_fraction': round(c1.longest_true_run(severe) / max(1, len(severe)), 6), 'image_found': image_path is not None, 'all_fragments': len(all_fragments), 'main_trunk_fragments': len(fragments), 'all_connection_candidates': len(candidates), 'main_trunk_edges': len(bridges), 'terminal_source': terminal_source, 'main_trunk_cost': round(trunk_cost, 6) if np.isfinite(trunk_cost) else '', 'maximum_selected_gap_px': round(maximum_gap, 4), 'maximum_selected_gap_widths': round(maximum_gap / max(road_width, 1.0), 6), 'selected_obstacle_support': round(max((float(row['obstacle_support']) for row in selected_rows), default=0.0), 6), 'selected_boundary_cross_fraction': round(max((float(row['boundary_cross_fraction']) for row in selected_rows), default=0.0), 6), 'error': ''}
    return (result, candidate_rows, slice_rows)

def error_result(item: dict, message: str) -> dict:
    """Error result."""
    result = {field: '' for field in RESULT_FIELDS}
    result.update({'file_name': item.get('key') or item.get('file_name') or item.get('name', ''), 'road_blockage_gt': item.get('road_blockage', ''), 'predicted_label': 'uncertain', 'correct': False, 'status': 'error', 'confidence': 'low', 'needs_review': True, 'error': message})
    return result

def main() -> int:
    """Run the complete dataset inference workflow."""
    args = parse_args()
    c1.validate_args(args)
    items = v1.load_items(args.json)
    if args.only:
        target = Path(args.only).stem.lower()
        items = [item for item in items if Path(str(item.get('key') or item.get('file_name') or item.get('name', ''))).stem.lower() == target]
        if not items:
            raise ValueError(f'找不到指定影像：{args.only}')
    if args.max_images is not None:
        items = items[:args.max_images]
    result_path = args.output_dir / 'centreline_graph_corridor_v1_2_main_trunk_results.csv'
    candidate_path = args.output_dir / 'main_trunk_candidates.csv'
    slice_path = args.output_dir / 'slice_profiles.csv'
    results = read_csv(result_path) if args.resume else []
    candidate_rows = read_csv(candidate_path) if args.resume else []
    slice_rows = read_csv(slice_path) if args.resume else []
    if args.resume:
        statuses = {str(row.get('file_name', '')): str(row.get('status', '')) for row in results}
        completed = {name for name, status in statuses.items() if not (args.retry_errors and status == 'error')}
        pending = [item for item in items if str(item.get('key') or item.get('file_name') or item.get('name', '')) not in completed]
        pending_names = {str(item.get('key') or item.get('file_name') or item.get('name', '')) for item in pending}
        results = [row for row in results if str(row.get('file_name', '')) not in pending_names]
        candidate_rows = [row for row in candidate_rows if str(row.get('file_name', '')) not in pending_names]
        slice_rows = [row for row in slice_rows if str(row.get('file_name', '')) not in pending_names]
        print(f'resume: 已保留 {len(completed)} 張，待處理 {len(pending)} 張', flush=True)
        items = pending
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        key = str(item.get('key') or item.get('file_name') or item.get('name', f'item_{index}'))
        print(f'[{index}/{len(items)}] {key}', flush=True)
        try:
            result, candidates, slices = process_item(item, args)
            print(f"  {result['predicted_label']} trunk={result['main_trunk_fragments']}/{result['all_fragments']} edges={result['main_trunk_edges']} max_gap_W={result['maximum_selected_gap_widths']} slices={result['valid_slices']}", flush=True)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__
            print(f'  ERROR: {message}', flush=True)
            result, candidates, slices = (error_result(item, message), [], [])
        results.append(result)
        candidate_rows.extend(candidates)
        slice_rows.extend(slices)
        write_csv(result_path, results, RESULT_FIELDS)
        write_csv(candidate_path, candidate_rows, CANDIDATE_FIELDS)
        write_csv(slice_path, slice_rows, c1.SLICE_FIELDS)
        gc.collect()
    valid = [row for row in results if row.get('status') != 'error']
    accuracy = sum((str(row.get('correct', '')).lower() == 'true' for row in valid)) / max(1, len(valid))
    print(f'完成：valid={len(valid)} error={len(results) - len(valid)} accuracy={accuracy:.4f}')
    print(f'結果：{result_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
'''

__EMBEDDED_CENTRELINE_GRAPH_CORRIDOR_V1_3_SOURCE = r'''
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Optional
try:
    import cv2
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.spatial import cKDTree
    import reconstruct_road_v1 as v1
    import assess_centerline_graph_corridor_v1 as c1
    import assess_centreline_graph_corridor_v1_2_main_trunk as m2
except ImportError as exc:
    raise SystemExit(f'缺少套件或 companion script。請將 V1.3、V1.2、V1 與 reconstruct_road_v1.py 放在同一個 scripts/。\n原始錯誤：{exc}') from exc
METHOD = 'centreline_graph_corridor_v1_3_main_trunk_local_width_prior'
RESULT_FIELDS = [*m2.RESULT_FIELDS[:-1], 'compression_detected_slices', 'compression_fraction', 'minimum_observed_expected_width_ratio', 'two_sided_width_prior_fraction', 'maximum_unknown_ratio_p95', 'local_severe_run_fraction', 'error']
SLICE_FIELDS = [*c1.SLICE_FIELDS, 'observed_width_px', 'expected_width_px', 'observed_expected_width_ratio', 'compressed_width', 'inferred_width', 'two_sided_width_prior', 'confirmed_free_width_ratio', 'known_ratio', 'unknown_ratio', 'obstacle_active', 'evaluable']

def parse_args():
    """Parse command-line arguments."""
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--width-compression-ratio', type=float, default=0.72)
    extra.add_argument('--compression-min-run-widths', type=float, default=0.25)
    extra.add_argument('--healthy-obstacle-clearance-widths', type=float, default=0.85)
    extra.add_argument('--healthy-width-minimum-ratio', type=float, default=0.55)
    extra.add_argument('--healthy-width-maximum-ratio', type=float, default=1.8)
    extra.add_argument('--width-prior-minimum-samples', type=int, default=3)
    extra.add_argument('--width-prior-minimum-width-ratio', type=float, default=0.55)
    extra.add_argument('--width-prior-maximum-width-ratio', type=float, default=1.8)
    extra.add_argument('--compression-obstacle-proximity-widths', type=float, default=0.85)
    extra.add_argument('--roadline-centre-snap-widths', type=float, default=0.45)
    extra.add_argument('--boundary-centre-snap-widths', type=float, default=1.6)
    extra.add_argument('--minimum-known-slice-ratio', type=float, default=0.6)
    extra.add_argument('--maximum-clear-unknown-ratio', type=float, default=0.25)
    extra.add_argument('--obstacle-active-ratio', type=float, default=0.02)
    extra_values, remaining = extra.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = m2.parse_args()
    finally:
        sys.argv = original
    for name, value in vars(extra_values).items():
        setattr(args, name, value)
    return args

def validate_local_args(args) -> None:
    """Validate local args."""
    c1.validate_args(args)
    for name in ('width_compression_ratio', 'healthy_width_minimum_ratio', 'width_prior_minimum_width_ratio', 'minimum_known_slice_ratio', 'maximum_clear_unknown_ratio', 'obstacle_active_ratio'):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} 必須介於 0 與 1")
    if args.healthy_width_maximum_ratio <= args.healthy_width_minimum_ratio:
        raise ValueError('healthy width maximum 必須大於 minimum')
    if args.width_prior_maximum_width_ratio <= args.width_prior_minimum_width_ratio:
        raise ValueError('width prior maximum 必須大於 minimum')
    if args.width_prior_minimum_samples < 1:
        raise ValueError('--width-prior-minimum-samples 至少為 1')

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path: Path) -> list[dict]:
    """Read csv."""
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))

def json_default(value):
    """Json default."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def suppress_short_true_runs(values: np.ndarray, minimum: int) -> np.ndarray:
    """Suppress short true runs."""
    result = values.astype(bool).copy()
    start = None
    for index in range(len(result) + 1):
        active = index < len(result) and bool(result[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start < minimum:
                result[start:index] = False
            start = None
    return result

def sample_bool(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample bool."""
    xy = np.rint(points).astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    output = np.zeros(len(points), dtype=bool)
    output[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return output

def sample_float(image: np.ndarray, points: np.ndarray, fallback: float) -> np.ndarray:
    """Sample float."""
    xy = np.rint(points).astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < image.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < image.shape[0])
    output = np.full(len(points), fallback, dtype=np.float32)
    output[valid] = image[xy[valid, 1], xy[valid, 0]]
    return output

def build_roadline_tree(road: np.ndarray, roadline: np.ndarray, road_width: float, args) -> Optional[cKDTree]:
    """Build roadline tree."""
    paths = c1.centre_roadline_paths(road, roadline, road_width, args)
    if not paths:
        return None
    points = np.vstack([path for path in paths if len(path)]).astype(np.float32)
    if not len(points):
        return None
    return cKDTree(points)

def roadline_snap(tree: Optional[cKDTree], point: np.ndarray, maximum: float) -> tuple[Optional[np.ndarray], float]:
    """Roadline snap."""
    if tree is None:
        return (None, float('inf'))
    distance, index = tree.query(np.asarray(point, dtype=np.float32), k=1)
    if not np.isfinite(distance) or float(distance) > maximum:
        return (None, float(distance))
    return (np.asarray(tree.data[int(index)], dtype=np.float32), float(distance))

def preferred_boundary_distance(cues: dict[str, np.ndarray], point: np.ndarray, direction: np.ndarray, maximum: float) -> tuple[Optional[float], str]:
    """Preferred boundary distance."""
    for name in ('guardrail', 'roadside_wall', 'curb'):
        distance = c1.ray_first(cues[name], point, direction, maximum)
        if distance is not None:
            return (float(distance), name)
    return (None, 'none')

def interpolate_width_prior(raw_width: np.ndarray, healthy: np.ndarray, fallback: float, road_width: float, args) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate width prior."""
    indices = np.arange(len(raw_width), dtype=np.float32)
    valid = np.flatnonzero(healthy & np.isfinite(raw_width))
    lower = args.width_prior_minimum_width_ratio * road_width
    upper = args.width_prior_maximum_width_ratio * road_width
    if len(valid) >= args.width_prior_minimum_samples:
        samples = raw_width[valid].astype(np.float32)
        filter_size = min(7, len(samples))
        samples = median_filter(samples, size=max(1, filter_size))
        expected = np.interp(indices, valid.astype(np.float32), samples).astype(np.float32)
        two_sided = (indices >= float(valid[0])) & (indices <= float(valid[-1]))
    elif len(valid):
        expected = np.full(len(raw_width), float(np.median(raw_width[valid])), np.float32)
        two_sided = np.zeros(len(raw_width), dtype=bool)
    else:
        expected = np.full(len(raw_width), fallback, np.float32)
        two_sided = np.zeros(len(raw_width), dtype=bool)
    if len(expected) >= 3:
        size = min(9, len(expected) if len(expected) % 2 else len(expected) - 1)
        expected = median_filter(expected, size=max(1, size)).astype(np.float32)
    return (np.clip(expected, lower, upper), two_sided)

def expected_geometry(road: np.ndarray, obstacles: np.ndarray, obstacle_distance: np.ndarray, path: np.ndarray, fallback_width: float, road_width: float, cues: dict[str, np.ndarray], roadline_tree: Optional[cKDTree], args) -> dict:
    """Estimate corrected centre positions and expected local widths."""
    path = c1.resample(path, args.sample_step_px)
    normals = v1.path_normals(path)
    base_left, base_right, base_observed = c1.variable_widths(road, obstacles, path, road_width, cues, args)
    raw_left = np.full(len(path), np.nan, np.float32)
    raw_right = np.full(len(path), np.nan, np.float32)
    ray_maximum = max(24.0, args.width_ray_widths * road_width)
    on_road = sample_bool(road, path)
    on_obstacle = sample_bool(obstacles, path)
    for index, (point, normal) in enumerate(zip(path, normals)):
        if not on_road[index] or on_obstacle[index]:
            continue
        left = v1.ray_distance(road, point, normal, ray_maximum)
        right = v1.ray_distance(road, point, -normal, ray_maximum)
        total = left + right
        if left > 0 and right > 0 and (0.25 * road_width <= total <= 2.5 * road_width):
            raw_left[index], raw_right[index] = (left, right)
    raw_width = raw_left + raw_right
    raw_valid = np.isfinite(raw_width)
    distance_to_obstacle = sample_float(obstacle_distance, path, fallback=10.0 * road_width)
    healthy = raw_valid & (distance_to_obstacle >= args.healthy_obstacle_clearance_widths * road_width) & (raw_width >= args.healthy_width_minimum_ratio * road_width) & (raw_width <= args.healthy_width_maximum_ratio * road_width)
    healthy_fallback = float(fallback_width) if np.isfinite(fallback_width) else road_width
    if healthy.any():
        healthy_fallback = float(np.median(raw_width[healthy]))
    expected_width, two_sided = interpolate_width_prior(raw_width, healthy, healthy_fallback, road_width, args)
    ratio = np.full(len(path), np.nan, np.float32)
    ratio[raw_valid] = raw_width[raw_valid] / np.maximum(expected_width[raw_valid], 1.0)
    near_obstacle = distance_to_obstacle <= args.compression_obstacle_proximity_widths * road_width
    compressed = raw_valid & near_obstacle & (ratio < args.width_compression_ratio)
    minimum_run = max(2, int(math.ceil(args.compression_min_run_widths * road_width / max(args.sample_step_px, 1.0))))
    compressed = suppress_short_true_runs(compressed, minimum_run)
    inferred = ~raw_valid & near_obstacle
    special = compressed | inferred
    corrected_path = path.copy()
    left_width = base_left.astype(np.float32).copy()
    right_width = base_right.astype(np.float32).copy()
    centre_source = np.full(len(path), 'visible_skeleton', dtype=object)
    boundary_maximum = max(20.0, args.boundary_centre_snap_widths * road_width)
    roadline_maximum = args.roadline_centre_snap_widths * road_width
    minimum_side = args.minimum_side_width_ratio * road_width
    for index in np.flatnonzero(special):
        point = path[index].copy()
        normal = normals[index]
        target_width = float(expected_width[index])
        line_point, _ = roadline_snap(roadline_tree, point, roadline_maximum)
        if line_point is not None:
            point = line_point
            centre_source[index] = 'road_line'
        plus, plus_name = preferred_boundary_distance(cues, point, normal, boundary_maximum)
        minus, minus_name = preferred_boundary_distance(cues, point, -normal, boundary_maximum)
        plausible_plus = plus is not None and minimum_side <= plus <= 1.5 * target_width
        plausible_minus = minus is not None and minimum_side <= minus <= 1.5 * target_width
        if plausible_plus and plausible_minus:
            span = float(plus + minus)
            if 0.55 * target_width <= span <= 1.65 * target_width:
                corrected_path[index] = point + 0.5 * (plus - minus) * normal
                target_width = float(np.clip(0.65 * target_width + 0.35 * span, args.width_prior_minimum_width_ratio * road_width, args.width_prior_maximum_width_ratio * road_width))
                centre_source[index] = f'two_boundaries:{plus_name}+{minus_name}'
        elif line_point is None and plausible_plus:
            corrected_path[index] = point + (plus - 0.5 * target_width) * normal
            centre_source[index] = f'one_boundary:{plus_name}'
        elif line_point is None and plausible_minus:
            corrected_path[index] = point - (minus - 0.5 * target_width) * normal
            centre_source[index] = f'one_boundary:{minus_name}'
        left_width[index] = 0.5 * target_width
        right_width[index] = 0.5 * target_width
    if len(corrected_path) >= 5 and special.any():
        corrected_path[:, 0] = median_filter(corrected_path[:, 0], size=5)
        corrected_path[:, 1] = median_filter(corrected_path[:, 1], size=5)
    return {'path': corrected_path.astype(np.float32), 'raw_path': path.astype(np.float32), 'left': left_width.astype(np.float32), 'right': right_width.astype(np.float32), 'observed': base_observed.astype(bool), 'raw_width': raw_width.astype(np.float32), 'expected_width': expected_width.astype(np.float32), 'width_ratio': ratio.astype(np.float32), 'compressed': compressed.astype(bool), 'inferred': inferred.astype(bool), 'two_sided_prior': two_sided.astype(bool), 'centre_source': centre_source}

def reconstruct_with_local_width_prior(shape: tuple[int, int], road: np.ndarray, obstacles: np.ndarray, fragments: list[c1.Fragment], bridges: list[c1.Bridge], cues: dict[str, np.ndarray], road_width: float, args) -> tuple[np.ndarray, list[dict], dict]:
    """Reconstruct the trunk with a local healthy-width prior."""
    parent = list(range(len(fragments)))

    def find(index: int) -> int:
        """Find."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for bridge in bridges:
        first, second = (find(bridge.first_fragment), find(bridge.second_fragment))
        if first != second:
            parent[second] = first
    route_ids = {index: find(index) for index in range(len(fragments))}
    paths: list[tuple[str, int, np.ndarray, float]] = [(f'fragment_{value.fragment_id}', route_ids[value.fragment_id], value.path, value.median_width) for value in fragments]
    paths += [(f'bridge_{value.bridge_id}', route_ids[value.first_fragment], value.path, road_width) for value in bridges]
    if obstacles.any():
        obstacle_distance = cv2.distanceTransform((~obstacles).astype(np.uint8), cv2.DIST_L2, 3)
    else:
        obstacle_distance = np.full(shape, 10.0 * road_width, np.float32)
    roadline_tree = build_roadline_tree(road, cues['road line'], road_width, args)
    reconstructed = road.copy()
    geometries: list[dict] = []
    observed_values: list[bool] = []
    compressed_values: list[bool] = []
    inferred_values: list[bool] = []
    two_sided_values: list[bool] = []
    finite_ratios: list[float] = []
    for path_id, route_id, path, fallback_width in paths:
        if len(path) < 2:
            continue
        geometry = expected_geometry(road, obstacles, obstacle_distance, path, fallback_width, road_width, cues, roadline_tree, args)
        geometry['path_id'] = path_id
        geometry['route_id'] = int(route_id)
        corridor = v1.sweep_road_polygon(shape, geometry['path'], geometry['left'], geometry['right'])
        reconstructed |= corridor
        observed_values.extend(geometry['observed'].tolist())
        compressed_values.extend(geometry['compressed'].tolist())
        inferred_values.extend(geometry['inferred'].tolist())
        two_sided_values.extend(geometry['two_sided_prior'].tolist())
        finite = geometry['width_ratio'][np.isfinite(geometry['width_ratio'])]
        finite_ratios.extend(finite.astype(float).tolist())
        geometries.append(geometry)
    del obstacle_distance
    count = max(1, len(observed_values))
    stats = {'observed_fraction': float(np.mean(observed_values)) if observed_values else 0.0, 'compression_count': int(sum(compressed_values)), 'compression_fraction': float(sum(compressed_values) / count), 'inferred_count': int(sum(inferred_values)), 'two_sided_fraction': float(np.mean(two_sided_values)) if two_sided_values else 0.0, 'minimum_width_ratio': float(min(finite_ratios)) if finite_ratios else 1.0}
    return (reconstructed, geometries, stats)

def longest_true_ratio(values: np.ndarray) -> float:
    """Longest true ratio."""
    return c1.longest_true_run(values.astype(bool).tolist()) / max(1, len(values))

def assess_cluster_local_width(cluster_id: int, cluster: np.ndarray, observed_road: np.ndarray, all_obstacles: np.ndarray, geometries: list[dict], road_width: float, args, file_name: str) -> tuple[str, str, str, list[dict], list[dict], dict]:
    """Assess one obstacle cluster using expected local widths."""
    yx = np.argwhere(cluster)
    if not len(yx):
        return ('uncertain', 'low', 'empty_cluster', [], [], {})
    route_groups: dict[int, list[dict]] = {}
    for geometry in geometries:
        route_groups.setdefault(int(geometry['route_id']), []).append(geometry)
    route_results = []
    step = max(1, int(round(args.slice_step_widths * road_width / max(args.sample_step_px, 1.0))))
    for route_id, route_geometries in route_groups.items():
        spans, anchor, tangent, _ = c1.route_spans_interaction(route_geometries, cluster, road_width, args)
        if not spans or anchor is None or tangent is None:
            continue
        diagonal = math.hypot(float(np.ptp(yx[:, 1])), float(np.ptp(yx[:, 0])))
        half_length = min(6.0 * road_width, max(2.0 * road_width, 0.5 * diagonal))
        rows: list[dict] = []
        drawings: list[dict] = []
        for geometry in route_geometries:
            path = geometry['path']
            left, right = (geometry['left'], geometry['right'])
            normals = v1.path_normals(path)
            longitudinal = (path - anchor[None, :]) @ tangent
            lateral = np.abs((path - anchor[None, :]) @ np.asarray([-tangent[1], tangent[0]], dtype=np.float32))
            indices = np.flatnonzero((np.abs(longitudinal) <= half_length) & (lateral <= 2.5 * road_width))
            for index in indices[::step]:
                expected_width = float(left[index] + right[index])
                sample_count = max(21, int(round(expected_width)) + 1)
                offsets = np.linspace(-float(right[index]), float(left[index]), sample_count, dtype=np.float32)
                points = path[index][None, :] + offsets[:, None] * normals[index][None, :]
                x = np.rint(points[:, 0]).astype(int)
                y = np.rint(points[:, 1]).astype(int)
                inside = (x >= 0) & (x < cluster.shape[1]) & (y >= 0) & (y < cluster.shape[0])
                if float(inside.mean()) < 0.9:
                    continue
                occupied = np.zeros(sample_count, dtype=bool)
                any_obstacle = np.zeros(sample_count, dtype=bool)
                visible_road = np.zeros(sample_count, dtype=bool)
                occupied[inside] = cluster[y[inside], x[inside]]
                any_obstacle[inside] = all_obstacles[y[inside], x[inside]]
                visible_road[inside] = observed_road[y[inside], x[inside]]
                confirmed_free = visible_road & ~any_obstacle
                known = confirmed_free | any_obstacle
                unknown = inside & ~known
                obstacle_ratio = float(occupied.mean())
                free_ratio = c1.longest_true_run(confirmed_free.tolist()) / sample_count
                known_ratio = float(known.mean())
                unknown_ratio = float(unknown.mean())
                evaluable = known_ratio >= args.minimum_known_slice_ratio or obstacle_ratio >= args.severe_obstacle_ratio
                severe = evaluable and (obstacle_ratio >= args.severe_obstacle_ratio or free_ratio <= args.blocked_free_width_ratio)
                raw_width = float(geometry['raw_width'][index])
                width_ratio = float(geometry['width_ratio'][index])
                active = obstacle_ratio >= args.obstacle_active_ratio
                rows.append({'file_name': file_name, 'cluster_id': cluster_id, 'route_id': route_id, 'path_id': geometry['path_id'], 'slice_index': int(index), 'center_x': round(float(path[index, 0]), 3), 'center_y': round(float(path[index, 1]), 3), 'road_width_px': round(expected_width, 3), 'obstacle_ratio': round(obstacle_ratio, 6), 'maximum_free_width_ratio': round(free_ratio, 6), 'severe': bool(severe), 'observed_width_px': round(raw_width, 3) if np.isfinite(raw_width) else '', 'expected_width_px': round(float(geometry['expected_width'][index]), 3), 'observed_expected_width_ratio': round(width_ratio, 6) if np.isfinite(width_ratio) else '', 'compressed_width': bool(geometry['compressed'][index]), 'inferred_width': bool(geometry['inferred'][index]), 'two_sided_width_prior': bool(geometry['two_sided_prior'][index]), 'confirmed_free_width_ratio': round(free_ratio, 6), 'known_ratio': round(known_ratio, 6), 'unknown_ratio': round(unknown_ratio, 6), 'obstacle_active': bool(active), 'evaluable': bool(evaluable)})
                drawings.append({'first': points[0], 'second': points[-1], 'severe': bool(severe), 'ratio': obstacle_ratio})
        rows.sort(key=lambda row: (str(row['path_id']), int(row['slice_index'])))
        evaluable_rows = [row for row in rows if bool(row['evaluable'])]
        if len(evaluable_rows) < args.minimum_valid_slices:
            continue
        active_rows = [row for row in evaluable_rows if bool(row['obstacle_active'])]
        ratios = [float(row['obstacle_ratio']) for row in evaluable_rows]
        unknown_values = [float(row['unknown_ratio']) for row in evaluable_rows]
        if active_rows:
            active_severe = np.asarray([bool(row['severe']) for row in active_rows], dtype=bool)
            severe_fraction = float(active_severe.mean())
            path_runs: dict[str, list[bool]] = {}
            for row in active_rows:
                path_runs.setdefault(str(row['path_id']), []).append(bool(row['severe']))
            longest_run = max((c1.longest_true_run(values) for values in path_runs.values()), default=0)
            run_fraction = longest_run / max(1, len(active_rows))
        else:
            severe_fraction = 0.0
            run_fraction = 0.0
        unknown_p95 = float(np.percentile(unknown_values, 95)) if unknown_values else 1.0
        if active_rows and severe_fraction >= args.blocked_severe_slice_fraction and (run_fraction >= args.blocked_consecutive_fraction):
            label, confidence, reason = ('blocked', 'medium', 'local_expected_width_sustained_severe')
        elif max(ratios, default=0.0) < args.clear_maximum_obstacle_ratio and unknown_p95 <= args.maximum_clear_unknown_ratio:
            label, confidence, reason = ('clear', 'medium', 'visible_route_below_clear_threshold')
        elif active_rows and unknown_p95 <= 1.0 - args.minimum_known_slice_ratio:
            label, confidence, reason = ('passable_restricted', 'medium', 'obstacle_present_with_confirmed_continuous_free_width')
        else:
            label, confidence, reason = ('uncertain', 'low', 'expected_corridor_contains_excess_unknown_space')
        route_results.append({'label': label, 'confidence': confidence, 'reason': reason, 'rows': evaluable_rows, 'drawings': drawings, 'unknown_p95': unknown_p95, 'severe_run_fraction': run_fraction, 'severe_fraction': severe_fraction})
    if not route_results:
        return ('uncertain', 'low', 'no_evaluable_route_across_obstacle', [], [], {})
    certain_labels = {value['label'] for value in route_results if value['label'] != 'uncertain'}
    if args.route_agreement_required and len(certain_labels) > 1:
        return ('uncertain', 'low', 'evaluable_route_disagreement', [], [], {})
    if not certain_labels:
        selected = route_results
        label = 'uncertain'
    else:
        label = max(certain_labels, key=lambda value: c1.LABEL_RANK[value])
        selected = [value for value in route_results if value['label'] == label]
    rows = [row for value in selected for row in value['rows']]
    drawings = [row for value in selected for row in value['drawings']]
    confidence = 'high' if len(selected) > 1 and label != 'uncertain' else selected[0]['confidence']
    meta = {'unknown_p95': max((value['unknown_p95'] for value in selected)), 'severe_run_fraction': max((value['severe_run_fraction'] for value in selected)), 'severe_fraction': max((value['severe_fraction'] for value in selected))}
    return (label, confidence, selected[0]['reason'], rows, drawings, meta)

def make_overlay(image_path: Optional[Path], road: np.ndarray, obstacles: np.ndarray, reconstructed: np.ndarray, geometries: list[dict], bridges: list[c1.Bridge], drawings: list[dict], predicted: str, confidence: str, terminal_source: str, trunk_cost: float, args) -> np.ndarray:
    """Draw masks, centrelines, bridges and cross-sections."""
    overlay = c1.make_overlay(image_path, road, obstacles, reconstructed, geometries, bridges, drawings, predicted, confidence, args, draw_header=False)
    scale = overlay.shape[1] / road.shape[1]
    radius = max(1, int(round(max(overlay.shape[:2]) / 900)))
    for geometry in geometries:
        active = geometry['compressed'] | geometry['inferred']
        for point in geometry['path'][active][::2]:
            xy = tuple(np.rint(point * scale).astype(int))
            cv2.circle(overlay, xy, radius, (0, 140, 255), -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (0, 0), (min(overlay.shape[1] - 1, 760), min(overlay.shape[0] - 1, 52)), (0, 0, 0), -1)
    colour = {'clear': (60, 230, 80), 'passable_restricted': (0, 210, 255), 'blocked': (40, 40, 255), 'uncertain': (255, 90, 255)}[predicted]
    cv2.putText(overlay, f'centreline graph V1.3 local width prior: {predicted}', (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    cost_text = 'inf' if not np.isfinite(trunk_cost) else f'{trunk_cost:.2f}'
    cv2.putText(overlay, f'confidence={confidence} source={terminal_source} bridges={len(bridges)} cost={cost_text}', (9, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1, cv2.LINE_AA)
    return overlay

def process_item(item: dict, args):
    """Process one annotated image and return its results."""
    key = str(item.get('key') or item.get('file_name') or item.get('name'))
    stem = Path(key).stem
    road = v1.clean_road_mask(v1.rasterize_labels(item, {'road'}), args.min_road_area, args.closing_radius)
    if not road.any():
        raise ValueError('找不到有效 road polygon')
    obstacles = v1.rasterize_labels(item, c1.OBSTACLE_LABELS)
    cues = {name: v1.rasterize_labels(item, {name}) for name in c1.CUE_LABELS}
    all_fragments, road_width = c1.build_fragments(road, cues['road line'], args)
    if not all_fragments:
        raise ValueError('找不到有效道路中心線')
    boundary = cues['guardrail'] | cues['roadside_wall'] | cues['curb']
    candidates, candidate_rows, endpoint_map = m2.build_unlimited_candidates(all_fragments, cues, boundary, obstacles, road_width, args)
    selected_ids, selected_bridges_old, terminal_source, trunk_cost, start, goal = m2.search_main_trunk(all_fragments, candidates, endpoint_map, road.shape, road_width, args)
    fragments, bridges = m2.remap_trunk(all_fragments, selected_ids, selected_bridges_old)
    selected_candidate_ids = {int(value.bridge_id) for value in selected_bridges_old}
    for row in candidate_rows:
        row['file_name'] = key
        row['selected'] = int(row['candidate_id']) in selected_candidate_ids
    selected_rows = [row for row in candidate_rows if row['selected']]
    weak_long_connection = any((float(row['gap_widths']) > args.maximum_gap_widths and float(row['obstacle_support']) < 0.03 and (float(row['cue_support']) < 0.18) for row in selected_rows))
    boundary_violation = any((float(row['boundary_cross_fraction']) > args.maximum_boundary_cross_fraction for row in selected_rows))
    geometry_review = weak_long_connection or boundary_violation
    reconstructed, geometries, width_stats = reconstruct_with_local_width_prior(road.shape, road, obstacles, fragments, bridges, cues, road_width, args)
    clusters = c1.cluster_obstacles(obstacles, road_width, args)
    adjacency = args.road_adjacency_widths * road_width
    relevant = [value for value in clusters if c1.local_mask_distance(value, reconstructed, adjacency) <= adjacency]
    slice_rows: list[dict] = []
    drawings: list[dict] = []
    cluster_results = []
    for cluster_id, cluster in enumerate(relevant, start=1):
        label, confidence, reason, rows, local_drawings, meta = assess_cluster_local_width(cluster_id, cluster, road, obstacles, geometries, road_width, args, key)
        slice_rows.extend(rows)
        drawings.extend(local_drawings)
        cluster_results.append({'label': label, 'confidence': confidence, 'reason': reason, 'rows': rows, 'meta': meta})
    if not clusters:
        predicted, confidence, reason = ('clear', 'high', 'no_obstacle_cluster')
    elif not relevant:
        predicted, confidence, reason = ('uncertain', 'low', 'obstacle_without_main_trunk_interaction')
    else:
        certain = [value for value in cluster_results if value['label'] != 'uncertain']
        if not certain:
            predicted, confidence, reason = ('uncertain', 'low', 'main_trunk_lacks_evaluable_cross_sections')
        else:
            worst = max(certain, key=lambda value: c1.LABEL_RANK[value['label']])
            unresolved = any((value['label'] == 'uncertain' for value in cluster_results))
            if unresolved and worst['label'] != 'blocked':
                predicted, confidence, reason = ('uncertain', 'low', 'unresolved_main_trunk_cluster')
            else:
                predicted = worst['label']
                confidence = worst['confidence']
                reason = worst['reason']
    if predicted != 'uncertain' and geometry_review:
        confidence = 'low'
        reason += ';forced_main_trunk_requires_review'
    image_path = v1.find_image(args.images_dir, key)
    if args.save_masks:
        v1.unicode_imwrite(args.output_dir / 'masks' / f'{stem}.png', reconstructed.astype(np.uint8) * 255)
    if not args.no_overlays:
        overlay = make_overlay(image_path, road, obstacles, reconstructed, geometries, bridges, drawings, predicted, confidence, terminal_source, trunk_cost, args)
        c1.save_jpeg(args.output_dir / 'overlays' / f'{stem}.jpg', overlay, args.overlay_jpeg_quality)
        del overlay
    diagnostic_path = args.output_dir / 'diagnostics' / f'{stem}.json'
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostic_path.open('w', encoding='utf-8') as file:
        json.dump({'file_name': key, 'prediction': predicted, 'confidence': confidence, 'reason': reason, 'method': METHOD, 'road_width_px': float(road_width), 'terminal_source': terminal_source, 'terminal_start': [int(value) for value in start], 'terminal_goal': [int(value) for value in goal], 'main_trunk_cost': float(trunk_cost) if np.isfinite(trunk_cost) else None, 'width_prior': width_stats, 'all_fragments': [{'id': int(value.fragment_id), 'source': value.source, 'length_px': float(c1.path_length(value.path)), 'median_width_px': float(value.median_width)} for value in all_fragments], 'selected_fragment_ids': [int(value) for value in selected_ids], 'selected_connections': selected_rows, 'cluster_results': [{'label': value['label'], 'confidence': value['confidence'], 'reason': value['reason'], 'valid_slices': len(value['rows']), 'meta': value['meta']} for value in cluster_results]}, file, ensure_ascii=False, indent=2, default=json_default)
    road_length = sum((c1.path_length(value.path) for value in fragments))
    bridge_length = sum((c1.path_length(value.path) for value in bridges))
    ratios = [float(row['obstacle_ratio']) for row in slice_rows]
    free = [float(row['maximum_free_width_ratio']) for row in slice_rows]
    active_severe = np.asarray([bool(row['severe']) for row in slice_rows if bool(row['obstacle_active'])], dtype=bool)
    unknown = [float(row['unknown_ratio']) for row in slice_rows]
    gt = str(item.get('road_blockage', ''))
    maximum_gap = max((value.gap_px for value in bridges), default=0.0)
    result = {'file_name': key, 'road_blockage_gt': gt, 'predicted_label': predicted, 'correct': predicted == gt, 'status': 'needs_review' if predicted == 'uncertain' or geometry_review else 'ok', 'confidence': confidence, 'needs_review': predicted == 'uncertain' or geometry_review, 'road_components': len(c1.connected_masks(road, args.min_road_area)), 'centreline_fragments': len(fragments), 'accepted_bridges': len(bridges), 'rejected_bridges': 0, 'centreline_coverage': round(road_length / max(1.0, road_length + bridge_length), 6), 'width_observation_fraction': round(width_stats['observed_fraction'], 6), 'obstacle_clusters': len(clusters), 'relevant_clusters': len(relevant), 'valid_slices': len(slice_rows), 'maximum_blockage_p95': round(float(np.percentile(ratios, 95)), 6) if ratios else 0.0, 'minimum_free_width_ratio': round(min(free), 6) if free else 1.0, 'maximum_severe_run_fraction': round(longest_true_ratio(active_severe), 6) if len(active_severe) else 0.0, 'image_found': image_path is not None, 'all_fragments': len(all_fragments), 'main_trunk_fragments': len(fragments), 'all_connection_candidates': len(candidates), 'main_trunk_edges': len(bridges), 'terminal_source': terminal_source, 'main_trunk_cost': round(float(trunk_cost), 6) if np.isfinite(trunk_cost) else '', 'maximum_selected_gap_px': round(float(maximum_gap), 4), 'maximum_selected_gap_widths': round(float(maximum_gap) / max(road_width, 1.0), 6), 'selected_obstacle_support': round(max((float(row['obstacle_support']) for row in selected_rows), default=0.0), 6), 'selected_boundary_cross_fraction': round(max((float(row['boundary_cross_fraction']) for row in selected_rows), default=0.0), 6), 'compression_detected_slices': width_stats['compression_count'], 'compression_fraction': round(width_stats['compression_fraction'], 6), 'minimum_observed_expected_width_ratio': round(width_stats['minimum_width_ratio'], 6), 'two_sided_width_prior_fraction': round(width_stats['two_sided_fraction'], 6), 'maximum_unknown_ratio_p95': round(float(np.percentile(unknown, 95)), 6) if unknown else 1.0, 'local_severe_run_fraction': round(longest_true_ratio(active_severe), 6) if len(active_severe) else 0.0, 'error': ''}
    return (result, candidate_rows, slice_rows)

def error_result(item: dict, message: str) -> dict:
    """Error result."""
    result = {field: '' for field in RESULT_FIELDS}
    result.update({'file_name': item.get('key') or item.get('file_name') or item.get('name', ''), 'road_blockage_gt': item.get('road_blockage', ''), 'predicted_label': 'uncertain', 'correct': False, 'status': 'error', 'confidence': 'low', 'needs_review': True, 'error': message})
    return result

def main() -> int:
    """Run the complete dataset inference workflow."""
    args = parse_args()
    validate_local_args(args)
    items = v1.load_items(args.json)
    if args.only:
        target = Path(args.only).stem.lower()
        items = [item for item in items if Path(str(item.get('key') or item.get('file_name') or item.get('name', ''))).stem.lower() == target]
        if not items:
            raise ValueError(f'找不到指定影像：{args.only}')
    if args.max_images is not None:
        items = items[:args.max_images]
    result_path = args.output_dir / 'centreline_graph_corridor_v1_3_main_trunk_local_width_prior_results.csv'
    candidate_path = args.output_dir / 'main_trunk_candidates.csv'
    slice_path = args.output_dir / 'slice_profiles.csv'
    results = read_csv(result_path) if args.resume else []
    candidate_rows = read_csv(candidate_path) if args.resume else []
    slice_rows = read_csv(slice_path) if args.resume else []
    if args.resume:
        statuses = {str(row.get('file_name', '')): str(row.get('status', '')) for row in results}
        completed = {name for name, status in statuses.items() if not (args.retry_errors and status == 'error')}
        pending = [item for item in items if str(item.get('key') or item.get('file_name') or item.get('name', '')) not in completed]
        pending_names = {str(item.get('key') or item.get('file_name') or item.get('name', '')) for item in pending}
        results = [row for row in results if str(row.get('file_name', '')) not in pending_names]
        candidate_rows = [row for row in candidate_rows if str(row.get('file_name', '')) not in pending_names]
        slice_rows = [row for row in slice_rows if str(row.get('file_name', '')) not in pending_names]
        print(f'resume: 已保留 {len(completed)} 張，待處理 {len(pending)} 張', flush=True)
        items = pending
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        key = str(item.get('key') or item.get('file_name') or item.get('name', f'item_{index}'))
        print(f'[{index}/{len(items)}] {key}', flush=True)
        try:
            result, candidates, slices = process_item(item, args)
            print(f"  {result['predicted_label']} trunk={result['main_trunk_fragments']}/{result['all_fragments']} edges={result['main_trunk_edges']} compressed={result['compression_detected_slices']} slices={result['valid_slices']}", flush=True)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__
            print(f'  ERROR: {message}', flush=True)
            result, candidates, slices = (error_result(item, message), [], [])
        results.append(result)
        candidate_rows.extend(candidates)
        slice_rows.extend(slices)
        write_csv(result_path, results, RESULT_FIELDS)
        write_csv(candidate_path, candidate_rows, m2.CANDIDATE_FIELDS)
        write_csv(slice_path, slice_rows, SLICE_FIELDS)
        gc.collect()
    valid = [row for row in results if row.get('status') != 'error']
    accuracy = sum((str(row.get('correct', '')).lower() == 'true' for row in valid)) / max(1, len(valid))
    print(f'完成：valid={len(valid)} error={len(results) - len(valid)} accuracy={accuracy:.4f}')
    print(f'結果：{result_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
'''

_road_tools = _load_embedded_module("reconstruct_road_v1", _RECONSTRUCT_ROAD_V1_SOURCE)
_centreline_v1 = _load_embedded_module(
    "assess_centerline_graph_corridor_v1", _ASSESS_CENTERLINE_GRAPH_CORRIDOR_V1_SOURCE
)
_main_trunk_v12 = _load_embedded_module(
    "assess_centreline_graph_corridor_v1_2_main_trunk",
    _ASSESS_CENTRELINE_GRAPH_CORRIDOR_V1_2_MAIN_TRUNK_SOURCE,
)
_v13 = _load_embedded_module(
    "_embedded_centreline_graph_corridor_v1_3",
    __EMBEDDED_CENTRELINE_GRAPH_CORRIDOR_V1_3_SOURCE,
)

if __name__ == "__main__":
    raise SystemExit(_v13.main())
