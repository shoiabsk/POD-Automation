#!/usr/bin/env python3
"""
E57 Point Cloud Processing Pipeline
==================================

A comprehensive pipeline for processing E57 point cloud files with clustering,
bounding box computation, and data export capabilities.

Features:
- Multi-library E57 support (pye57, e57)
- Advanced preprocessing (voxel downsampling, outlier removal)
- Multiple clustering algorithms (K-means, DBSCAN)
- Automatic cluster count selection
- Oriented bounding box computation
- Multi-format export (Excel, PLY, PY7)
- Unit conversion (always exports to inches)

Author: [Your Name]
Date: August 19, 2025
Version: 2.0
License: MIT
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# CONFIGURATION
# =============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_VOXEL_SIZE = 0.02
DEFAULT_NB_NEIGHBORS = 20
DEFAULT_STD_RATIO = 2.0
DEFAULT_MIN_CLUSTER_POINTS = 200
DEFAULT_K_RANGE = (2, 12)
DEFAULT_DBSCAN_EPS = 0.05
DEFAULT_DBSCAN_MIN_SAMPLES = 10

# Unit conversion constants
MM_TO_INCHES = 1.0 / 25.4
INCHES_TO_MM = 25.4


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_unit_scale_to_inches(unit: str) -> float:
    """
    Get scale factor to convert from given unit to inches.

    Args:
        unit: Unit string ('mm', 'millimeters', 'in', 'inches', etc.)

    Returns:
        Scale factor to multiply values to get inches
    """
    unit_normalized = unit.strip().lower()

    unit_mappings = {
        'mm': MM_TO_INCHES,
        'millimeter': MM_TO_INCHES,
        'millimeters': MM_TO_INCHES,
        'in': 1.0,
        'inch': 1.0,
        'inches': 1.0,
    }

    return unit_mappings.get(unit_normalized, 1.0)


def convert_linear_measurements_to_inches(data: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """
    Convert linear measurements in a data dictionary to inches.

    Args:
        data: Dictionary containing measurements
        scale: Scale factor to apply to linear measurements

    Returns:
        New dictionary with converted measurements
    """
    converted = data.copy()

    # Linear measurement fields
    linear_fields = ['center_x', 'center_y', 'center_z', 'length', 'width', 'height']

    for field in linear_fields:
        if field in converted:
            converted[field] *= scale

    return converted


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.1f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m {seconds % 60:.1f}s"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class E57ProcessingError(Exception):
    """Base exception for E57 processing pipeline."""
    pass


class E57LoadError(E57ProcessingError):
    """Exception raised when E57 file loading fails."""
    pass


class ClusteringError(E57ProcessingError):
    """Exception raised during clustering operations."""
    pass


class ExportError(E57ProcessingError):
    """Exception raised during data export."""
    pass


# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

class E57Dependencies:
    """Manages E57 library dependencies and provides unified interface."""

    def __init__(self):
        self.pye57_module = None
        self.e57_module = None
        self._initialize_libraries()

    def _initialize_libraries(self) -> None:
        """Initialize available E57 libraries."""
        # Try pye57 first (preferred)
        try:
            import pye57
            self.pye57_module = pye57
            logger.info("✓ pye57 library loaded successfully")
        except ImportError:
            logger.warning("✗ pye57 library not available")

        # Try e57 as fallback
        if not self.pye57_module:
            try:
                import e57
                self.e57_module = e57
                logger.info("✓ e57 library loaded successfully")
            except ImportError:
                logger.warning("✗ e57 library not available")

    @property
    def has_support(self) -> bool:
        """Check if any E57 library is available."""
        return self.pye57_module is not None or self.e57_module is not None

    @property
    def preferred_library(self) -> str:
        """Get name of preferred/available library."""
        if self.pye57_module:
            return "pye57"
        elif self.e57_module:
            return "e57"
        return "none"


# Global dependency manager instance
e57_deps = E57Dependencies()


# =============================================================================
# DATA MODELS
# =============================================================================

class PointCloudData:
    """Container for point cloud data with validation."""

    def __init__(
            self,
            points: np.ndarray,
            colors: Optional[np.ndarray] = None,
            intensity: Optional[np.ndarray] = None
    ):
        self.points = points
        self.colors = colors
        self.intensity = intensity
        self._validate()

    def _validate(self) -> None:
        """Validate data consistency and integrity."""
        if self.points is None or len(self.points) == 0:
            raise ValueError("Points array cannot be empty")

        if self.points.shape[1] != 3:
            raise ValueError(f"Points must have 3 coordinates, got {self.points.shape[1]}")

        n_points = len(self.points)

        if self.colors is not None:
            if len(self.colors) != n_points:
                raise ValueError(f"Colors array length mismatch: {len(self.colors)} != {n_points}")
            if self.colors.shape[1] != 3:
                raise ValueError(f"Colors must have 3 channels (RGB), got {self.colors.shape[1]}")

        if self.intensity is not None and len(self.intensity) != n_points:
            raise ValueError(f"Intensity array length mismatch: {len(self.intensity)} != {n_points}")

    @property
    def point_count(self) -> int:
        """Get number of points."""
        return len(self.points)

    @property
    def has_colors(self) -> bool:
        """Check if color data is available."""
        return self.colors is not None

    @property
    def has_intensity(self) -> bool:
        """Check if intensity data is available."""
        return self.intensity is not None

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box as (min_coords, max_coords)."""
        return np.min(self.points, axis=0), np.max(self.points, axis=0)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        min_coords, max_coords = self.bounds
        return {
            'point_count': self.point_count,
            'has_colors': self.has_colors,
            'has_intensity': self.has_intensity,
            'bounds_min': min_coords.tolist(),
            'bounds_max': max_coords.tolist(),
            'extent': (max_coords - min_coords).tolist()
        }


class ClusterInfo:
    """Container for cluster analysis results."""

    def __init__(
            self,
            asset_id: int,
            cluster_id: int,
            center: np.ndarray,
            dimensions: np.ndarray,
            rotation: np.ndarray,
            point_count: int = 0
    ):
        self.asset_id = asset_id
        self.cluster_id = cluster_id
        self.center = np.asarray(center)
        self.dimensions = np.asarray(dimensions)  # [length, width, height]
        self.rotation = np.asarray(rotation)  # [rx, ry, rz] in degrees
        self.point_count = point_count

    @property
    def volume(self) -> float:
        """Calculate approximate volume (length × width × height)."""
        return np.prod(self.dimensions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'asset_id': self.asset_id,
            'cluster_id': self.cluster_id,
            'center_x': float(self.center[0]),
            'center_y': float(self.center[1]),
            'center_z': float(self.center[2]),
            'length': float(self.dimensions),
            'width': float(self.dimensions[1]),
            'height': float(self.dimensions[2]),
            'rot_x': float(self.rotation),
            'rot_y': float(self.rotation[1]),
            'rot_z': float(self.rotation[2]),
            'volume': float(self.volume),
            'point_count': self.point_count
        }


# =============================================================================
# CORE COMPONENTS
# =============================================================================

class E57Loader:
    """Handles loading E57 files with multiple library backends."""

    def __init__(self):
        if not e57_deps.has_support:
            raise E57LoadError(
                "No E57 library available. Please install one of:\n"
                "  pip install pye57\n"
                "  pip install e57"
            )

    def load(
            self,
            file_path: str,
            load_colors: bool = True,
            load_intensity: bool = True
    ) -> PointCloudData:
        """
        Load point cloud data from E57 file.

        Args:
            file_path: Path to E57 file
            load_colors: Whether to load color data
            load_intensity: Whether to load intensity data

        Returns:
            PointCloudData object

        Raises:
            E57LoadError: If loading fails
        """
        if not os.path.exists(file_path):
            raise E57LoadError(f"E57 file not found: {file_path}")

        logger.info(f"Loading E57 file: {file_path}")
        logger.info(f"Using library: {e57_deps.preferred_library}")

        try:
            if e57_deps.pye57_module:
                return self._load_with_pye57(file_path, load_colors, load_intensity)
            elif e57_deps.e57_module:
                return self._load_with_e57(file_path, load_colors, load_intensity)
            else:
                raise E57LoadError("No E57 library available")

        except Exception as e:
            raise E57LoadError(f"Failed to load E57 file: {e}")

    def _load_with_pye57(
            self,
            file_path: str,
            load_colors: bool,
            load_intensity: bool
    ) -> PointCloudData:
        """Load using pye57 library."""
        e57_file = e57_deps.pye57_module.E57(file_path)
        scan_count = e57_file.scan_count
        logger.info(f"Found {scan_count} scans in E57 file")

        all_points = []
        all_colors = []
        all_intensity = []

        for scan_idx in range(scan_count):
            logger.info(f"Processing scan {scan_idx + 1}/{scan_count}")

            try:
                scan_data = e57_file.read_scan(
                    scan_idx,
                    intensity=load_intensity,
                    colors=load_colors
                )

                # Extract coordinates
                x = scan_data.get("cartesianX")
                y = scan_data.get("cartesianY")
                z = scan_data.get("cartesianZ")

                if any(coord is None for coord in (x, y, z)):
                    logger.warning(f"Scan {scan_idx}: missing coordinate data, skipping")
                    continue

                if any(len(coord) == 0 for coord in (x, y, z)):
                    logger.warning(f"Scan {scan_idx}: empty coordinate arrays, skipping")
                    continue

                points = np.column_stack([x, y, z])
                all_points.append(points)

                # Extract colors if requested and available
                if load_colors and all(
                        key in scan_data for key in ["colorRed", "colorGreen", "colorBlue"]
                ):
                    colors = np.column_stack([
                        np.asarray(scan_data["colorRed"]) / 255.0,
                        np.asarray(scan_data["colorGreen"]) / 255.0,
                        np.asarray(scan_data["colorBlue"]) / 255.0
                    ])
                    all_colors.append(colors)

                # Extract intensity if requested and available
                if load_intensity and "intensity" in scan_data and scan_data["intensity"] is not None:
                    all_intensity.append(np.asarray(scan_data["intensity"]))

            except Exception as e:
                logger.warning(f"Error processing scan {scan_idx}: {e}")
                continue

        if not all_points:
            raise E57LoadError("No valid point data found in E57 file")

        # Combine all scans
        points = np.vstack(all_points)
        colors = np.vstack(all_colors) if all_colors else None
        intensity = np.concatenate(all_intensity) if all_intensity else None

        logger.info(f"Loaded {len(points):,} points from {len(all_points)} scans")
        return PointCloudData(points, colors, intensity)

    def _load_with_e57(
            self,
            file_path: str,
            load_colors: bool,
            load_intensity: bool
    ) -> PointCloudData:
        """Load using e57 library."""
        point_cloud = e57_deps.e57_module.read_points(file_path)

        points = np.asarray(point_cloud.points)
        colors = None
        intensity = None

        if load_colors and hasattr(point_cloud, 'color') and point_cloud.color is not None:
            colors = np.asarray(point_cloud.color)

        if load_intensity and hasattr(point_cloud, 'intensity') and point_cloud.intensity is not None:
            intensity = np.asarray(point_cloud.intensity)

        logger.info(f"Loaded {len(points):,} points")
        return PointCloudData(points, colors, intensity)


class PointCloudPreprocessor:
    """Handles point cloud preprocessing operations."""

    def __init__(
            self,
            voxel_size: float = DEFAULT_VOXEL_SIZE,
            nb_neighbors: int = DEFAULT_NB_NEIGHBORS,
            std_ratio: float = DEFAULT_STD_RATIO
    ):
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def process(self, point_data: PointCloudData) -> o3d.geometry.PointCloud:
        """
        Preprocess point cloud data.

        Args:
            point_data: Input point cloud data

        Returns:
            Preprocessed Open3D PointCloud
        """
        logger.info("Starting point cloud preprocessing...")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_data.points)

        if point_data.has_colors:
            pcd.colors = o3d.utility.Vector3dVector(point_data.colors)

        original_count = len(pcd.points)
        logger.info(f"Original point count: {original_count:,}")

        # Voxel downsampling
        if self.voxel_size > 0:
            logger.info(f"Applying voxel downsampling (voxel_size={self.voxel_size})")
            pcd = pcd.voxel_down_sample(self.voxel_size)
            after_voxel = len(pcd.points)
            logger.info(f"After voxel downsampling: {after_voxel:,} points "
                        f"({100 * after_voxel / original_count:.1f}% retained)")

        # Statistical outlier removal
        logger.info(f"Applying statistical outlier removal "
                    f"(nb_neighbors={self.nb_neighbors}, std_ratio={self.std_ratio})")
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio
        )

        final_count = len(pcd.points)
        reduction_percent = 100 * (original_count - final_count) / original_count

        logger.info(f"After outlier removal: {final_count:,} points")
        logger.info(f"Total reduction: {original_count - final_count:,} points ({reduction_percent:.1f}%)")

        return pcd


class ClusterAnalyzer:
    """Handles point cloud clustering operations."""

    def __init__(self, method: str = "kmeans"):
        self.method = method.lower()
        if self.method not in ["kmeans", "dbscan"]:
            raise ValueError(f"Unsupported clustering method: {method}")

    def auto_select_k(
            self,
            points: np.ndarray,
            k_min: int = DEFAULT_K_RANGE[0],
            k_max: int = DEFAULT_K_RANGE[1]
    ) -> int:
        """
        Automatically select optimal number of clusters using silhouette analysis.

        Args:
            points: Point coordinates
            k_min: Minimum number of clusters to test
            k_max: Maximum number of clusters to test

        Returns:
            Optimal number of clusters
        """
        logger.info(f"Auto-selecting k using silhouette analysis (k={k_min}-{k_max})")

        best_k = k_min
        best_score = -1
        scores = {}

        max_k = min(k_max, len(points) - 1)

        for k in range(k_min, max_k + 1):
            try:
                logger.info(f"Testing k={k}...")

                kmeans = KMeans(
                    n_clusters=k,
                    n_init="auto",
                    random_state=42,
                    max_iter=300
                )
                labels = kmeans.fit_predict(points)

                if len(np.unique(labels)) < 2:
                    logger.warning(f"k={k}: Less than 2 unique clusters, skipping")
                    continue

                score = silhouette_score(points, labels)
                scores[k] = score

                logger.info(f"k={k}: silhouette score = {score:.4f}")

                if score > best_score:
                    best_k = k
                    best_score = score

            except Exception as e:
                logger.warning(f"Failed to evaluate k={k}: {e}")
                continue

        logger.info(f"Selected k={best_k} with silhouette score {best_score:.4f}")
        return best_k

    def cluster_kmeans(self, points: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Perform K-means clustering.

        Args:
            points: Point coordinates
            k: Number of clusters (None for auto-selection)

        Returns:
            Cluster labels
        """
        if k is None or k <= 0:
            k = self.auto_select_k(points)

        logger.info(f"Performing K-means clustering with k={k}")

        try:
            kmeans = KMeans(
                n_clusters=k,
                n_init="auto",
                random_state=42,
                max_iter=300
            )
            labels = kmeans.fit_predict(points)

            unique_labels = len(np.unique(labels))
            logger.info(f"K-means completed: {unique_labels} clusters created")

            return labels

        except Exception as e:
            raise ClusteringError(f"K-means clustering failed: {e}")

    def cluster_dbscan(
            self,
            points: np.ndarray,
            eps: float = DEFAULT_DBSCAN_EPS,
            min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering.

        Args:
            points: Point coordinates
            eps: Maximum distance between points in same neighborhood
            min_samples: Minimum points required to form dense region

        Returns:
            Cluster labels (-1 for noise points)
        """
        logger.info(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})")

        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(points)

            unique_labels = len(np.unique(labels[labels >= 0]))
            noise_points = np.sum(labels == -1)
            noise_percent = 100 * noise_points / len(points)

            logger.info(f"DBSCAN completed: {unique_labels} clusters, "
                        f"{noise_points} noise points ({noise_percent:.1f}%)")

            return labels

        except Exception as e:
            raise ClusteringError(f"DBSCAN clustering failed: {e}")

    def cluster(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform clustering using the configured method.

        Args:
            points: Point coordinates
            **kwargs: Method-specific parameters

        Returns:
            Cluster labels
        """
        if self.method == "kmeans":
            return self.cluster_kmeans(points, kwargs.get("k"))
        elif self.method == "dbscan":
            return self.cluster_dbscan(
                points,
                kwargs.get("eps", DEFAULT_DBSCAN_EPS),
                kwargs.get("min_samples", DEFAULT_DBSCAN_MIN_SAMPLES)
            )
        else:
            raise ClusteringError(f"Unknown clustering method: {self.method}")


class GeometryAnalyzer:
    """Handles geometric analysis of point clusters."""

    @staticmethod
    def rotation_matrix_to_euler_xyz(rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to Euler angles (XYZ order).

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            Euler angles [rx, ry, rz] in degrees
        """
        # Clamp values to avoid numerical issues
        r20 = np.clip(rotation_matrix[2, 0], -1.0, 1.0)
        ry = math.asin(-r20)
        cy = math.cos(ry)

        if abs(cy) > 1e-8:
            rx = math.atan2(rotation_matrix[2, 1] / cy, rotation_matrix[2, 2] / cy)
            rz = math.atan2(rotation_matrix[1, 0] / cy, rotation_matrix[0, 0] / cy)
        else:
            rx = 0.0
            rz = math.atan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

        return np.degrees([rx, ry, rz])

    def analyze_single_cluster(
            self,
            points: np.ndarray,
            cluster_id: int
    ) -> Optional[ClusterInfo]:
        """
        Analyze a single cluster and compute its properties.

        Args:
            points: Points belonging to the cluster
            cluster_id: Cluster identifier

        Returns:
            ClusterInfo object or None if analysis fails
        """
        try:
            # Create Open3D point cloud for the cluster
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(points)

            # Compute axis-aligned bounding box as fallback
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            center_fallback = aabb.get_center()
            extent_fallback = np.array(aabb.get_extent())

            # Attempt to compute oriented bounding box
            center = center_fallback
            extent = extent_fallback
            rotation_angles = np.zeros(3)

            try:
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(cluster_pcd.points)
                center = obb.get_center()
                extent = np.array(obb.extent)
                rotation_matrix = np.asarray(obb.R)
                rotation_angles = self.rotation_matrix_to_euler_xyz(rotation_matrix)

            except Exception as e:
                logger.warning(
                    f"OBB computation failed for cluster {cluster_id}: {e}. "
                    "Using axis-aligned bounding box."
                )

            # Sort dimensions: length >= width >= height
            dimensions_sorted = np.sort(extent)[::-1]

            return ClusterInfo(
                asset_id=cluster_id + 1,
                cluster_id=cluster_id + 1,
                center=center,
                dimensions=dimensions_sorted,
                rotation=rotation_angles,
                point_count=len(points)
            )

        except Exception as e:
            logger.error(f"Failed to analyze cluster {cluster_id}: {e}")
            return None

    def analyze_clusters(
            self,
            point_cloud: o3d.geometry.PointCloud,
            labels: np.ndarray,
            min_points: int = DEFAULT_MIN_CLUSTER_POINTS
    ) -> List[ClusterInfo]:
        """
        Analyze all clusters and compute their geometric properties.

        Args:
            point_cloud: Preprocessed point cloud
            labels: Cluster labels for each point
            min_points: Minimum points required for valid cluster

        Returns:
            List of ClusterInfo objects
        """
        logger.info("Analyzing cluster geometries...")

        points = np.asarray(point_cloud.points)
        results = []

        unique_labels = np.unique(labels)
        valid_clusters = 0
        skipped_clusters = 0

        for cluster_id in unique_labels:
            if cluster_id < 0:  # Skip noise points (DBSCAN)
                continue

            # Extract cluster points
            cluster_mask = labels == cluster_id
            cluster_points = points[cluster_mask]

            if len(cluster_points) < min_points:
                logger.warning(
                    f"Cluster {cluster_id}: {len(cluster_points)} points "
                    f"(< {min_points} required), skipping"
                )
                skipped_clusters += 1
                continue

            # Analyze cluster
            cluster_info = self.analyze_single_cluster(cluster_points, cluster_id)

            if cluster_info is not None:
                results.append(cluster_info)
                valid_clusters += 1
                logger.debug(
                    f"Cluster {cluster_id}: {len(cluster_points)} points, "
                    f"dimensions: {cluster_info.dimensions}"
                )

        logger.info(
            f"Cluster analysis complete: {valid_clusters} valid clusters, "
            f"{skipped_clusters} skipped (too few points)"
        )

        return results


class DataExporter:
    """Handles export of processed data to various formats."""

    @staticmethod
    def export_excel(
            cluster_results: List[ClusterInfo],
            metadata: Dict[str, Any],
            output_path: str
    ) -> None:
        """
        Export cluster results and metadata to Excel (measurements in inches).

        Args:
            cluster_results: List of cluster analysis results
            metadata: Processing metadata
            output_path: Output Excel file path

        Raises:
            ExportError: If export fails
        """
        logger.info(f"Exporting results to Excel: {output_path}")

        try:
            # Determine unit conversion
            cloud_units = metadata.get("cloud_units", "in")
            scale_to_inches = get_unit_scale_to_inches(cloud_units)

            logger.info(f"Converting from {cloud_units} to inches (scale: {scale_to_inches:.6f})")

            # Convert cluster data to inches
            cluster_data = []
            for cluster in cluster_results:
                cluster_dict = cluster.to_dict()
                cluster_dict_inches = convert_linear_measurements_to_inches(
                    cluster_dict, scale_to_inches
                )
                cluster_data.append(cluster_dict_inches)

            df_clusters = pd.DataFrame(cluster_data)

            # Prepare metadata with unit conversion
            metadata_export = metadata.copy()
            metadata_export["export_units"] = "inches"
            metadata_export["original_units"] = cloud_units
            metadata_export["unit_scale_factor"] = scale_to_inches

            # Convert voxel size to inches for reporting
            if "voxel_size" in metadata_export and isinstance(metadata_export["voxel_size"], (int, float)):
                metadata_export["voxel_size_inches"] = metadata_export["voxel_size"] * scale_to_inches

            df_metadata = pd.DataFrame([metadata_export])

            # Write to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_clusters.to_excel(writer, sheet_name='Clusters', index=False)
                df_metadata.to_excel(writer, sheet_name='Metadata', index=False)

                # Format the clusters sheet
                clusters_worksheet = writer.sheets['Clusters']

                # Auto-adjust column widths
                for column in clusters_worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter

                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass

                    adjusted_width = min(max_length + 2, 20)
                    clusters_worksheet.column_dimensions[column_letter].width = adjusted_width

            logger.info(f"Successfully exported {len(cluster_results)} clusters to Excel")

        except Exception as e:
            raise ExportError(f"Failed to export Excel: {e}")

    @staticmethod
    def export_ply(point_data: PointCloudData, output_path: str) -> None:
        """
        Export point cloud to PLY format.

        Args:
            point_data: Point cloud data to export
            output_path: Output PLY file path

        Raises:
            ExportError: If export fails
        """
        logger.info(f"Exporting point cloud to PLY: {output_path}")

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_data.points)

            if point_data.has_colors:
                pcd.colors = o3d.utility.Vector3dVector(point_data.colors)

            success = o3d.io.write_point_cloud(output_path, pcd)

            if not success:
                raise ExportError("Open3D failed to write PLY file")

            logger.info(f"Successfully exported {point_data.point_count:,} points to PLY")

        except Exception as e:
            raise ExportError(f"Failed to export PLY: {e}")

    @staticmethod
    def export_py7(
            point_data: PointCloudData,
            output_path: str,
            source_filename: str
    ) -> None:
        """
        Export to custom PY7 format (compressed NumPy).

        Args:
            point_data: Point cloud data to export
            output_path: Output PY7 file path
            source_filename: Original source filename

        Raises:
            ExportError: If export fails
        """
        logger.info(f"Exporting to PY7 format: {output_path}")

        try:
            # Create metadata
            export_metadata = {
                "source_file": source_filename,
                "export_timestamp": datetime.utcnow().isoformat() + "Z",
                "point_count": point_data.point_count,
                "has_colors": point_data.has_colors,
                "has_intensity": point_data.has_intensity,
                "format_version": "2.0",
                "bounds": point_data.summary()["bounds_min"] + point_data.summary()["bounds_max"]
            }

            # Save compressed
            np.savez_compressed(
                output_path,
                points=point_data.points,
                colors=point_data.colors,
                intensity=point_data.intensity,
                metadata=json.dumps(export_metadata, indent=2)
            )

            logger.info("Successfully exported to PY7 format")

        except Exception as e:
            raise ExportError(f"Failed to export PY7: {e}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class E57ProcessingPipeline:
    """Main processing pipeline that orchestrates all operations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        # Core components
        self.loader = E57Loader()

        self.preprocessor = PointCloudPreprocessor(
            voxel_size=self.config.get('voxel_size', DEFAULT_VOXEL_SIZE),
            nb_neighbors=self.config.get('nb_neighbors', DEFAULT_NB_NEIGHBORS),
            std_ratio=self.config.get('std_ratio', DEFAULT_STD_RATIO)
        )

        self.clusterer = ClusterAnalyzer(
            method=self.config.get('cluster_method', 'kmeans')
        )

        self.geometry_analyzer = GeometryAnalyzer()
        self.exporter = DataExporter()

        logger.info(f"Pipeline initialized with method: {self.clusterer.method}")

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete processing pipeline.

        Returns:
            Dictionary containing processing results and statistics
        """
        start_time = time.time()
        pipeline_start = datetime.now()

        logger.info("=" * 80)
        logger.info("E57 POINT CLOUD PROCESSING PIPELINE STARTING")
        logger.info("=" * 80)
        logger.info(f"Start time: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Input file: {self.config['input_file']}")

        try:
            # Stage 1: Load E57 file
            logger.info("\n[STAGE 1/7] Loading E57 file...")
            stage_start = time.time()

            point_data = self.loader.load(self.config['input_file'])
            logger.info(f"Stage 1 completed in {time.time() - stage_start:.1f}s")
            logger.info(f"Loaded: {point_data.summary()}")

            # Stage 2: Optional raw data exports
            if self.config.get('ply_output') or self.config.get('py7_output'):
                logger.info("\n[STAGE 2/7] Exporting raw data...")
                stage_start = time.time()

                if self.config.get('ply_output'):
                    self.exporter.export_ply(point_data, self.config['ply_output'])

                if self.config.get('py7_output'):
                    self.exporter.export_py7(
                        point_data,
                        self.config['py7_output'],
                        os.path.basename(self.config['input_file'])
                    )

                logger.info(f"Stage 2 completed in {time.time() - stage_start:.1f}s")
            else:
                logger.info("\n[STAGE 2/7] Skipping raw data exports (not requested)")

            # Stage 3: Preprocessing
            logger.info("\n[STAGE 3/7] Preprocessing point cloud...")
            stage_start = time.time()

            processed_pcd = self.preprocessor.process(point_data)

            logger.info(f"Stage 3 completed in {time.time() - stage_start:.1f}s")

            # Stage 4: Clustering
            logger.info("\n[STAGE 4/7] Performing clustering...")
            stage_start = time.time()

            points_array = np.asarray(processed_pcd.points)

            cluster_params = {
                'k': self.config.get('k', 0),
                'eps': self.config.get('eps', DEFAULT_DBSCAN_EPS),
                'min_samples': self.config.get('min_samples', DEFAULT_DBSCAN_MIN_SAMPLES)
            }

            labels = self.clusterer.cluster(points_array, **cluster_params)

            logger.info(f"Stage 4 completed in {time.time() - stage_start:.1f}s")

            # Stage 5: Geometric analysis
            logger.info("\n[STAGE 5/7] Analyzing cluster geometry...")
            stage_start = time.time()

            cluster_results = self.geometry_analyzer.analyze_clusters(
                processed_pcd,
                labels,
                min_points=self.config.get('min_cluster_points', DEFAULT_MIN_CLUSTER_POINTS)
            )

            logger.info(f"Stage 5 completed in {time.time() - stage_start:.1f}s")

            # Stage 6: Create processing metadata
            logger.info("\n[STAGE 6/7] Generating metadata...")

            total_processing_time = time.time() - start_time

            metadata = {
                # Input information
                'input_file': os.path.basename(self.config['input_file']),
                'input_file_size_mb': os.path.getsize(self.config['input_file']) / (1024 * 1024),

                # Processing information
                'processing_start': pipeline_start.isoformat(),
                'processing_time_seconds': total_processing_time,
                'processing_time_formatted': format_duration(total_processing_time),

                # Point counts
                'total_points_original': point_data.point_count,
                'total_points_processed': len(points_array),
                'point_reduction_percent': 100 * (1 - len(points_array) / point_data.point_count),

                # Clustering results
                'clusters_found': len(cluster_results),
                'clustering_method': self.config.get('cluster_method', 'kmeans'),

                # Processing parameters
                'voxel_size': self.config.get('voxel_size', DEFAULT_VOXEL_SIZE),
                'nb_neighbors': self.config.get('nb_neighbors', DEFAULT_NB_NEIGHBORS),
                'std_ratio': self.config.get('std_ratio', DEFAULT_STD_RATIO),
                'min_cluster_points': self.config.get('min_cluster_points', DEFAULT_MIN_CLUSTER_POINTS),

                # Data characteristics
                'has_colors': point_data.has_colors,
                'has_intensity': point_data.has_intensity,

                # Software information
                'pipeline_version': '2.0',
                'library_used': e57_deps.preferred_library
            }

            # Set native point cloud units (modify this line as needed)
            # Set to 'mm' if your E57 coordinates are in millimeters
            # Set to 'in' if your E57 coordinates are in inches
            metadata['cloud_units'] = 'mm'

            logger.info("Stage 6 completed")

            # Stage 7: Export results
            logger.info("\n[STAGE 7/7] Exporting results...")
            stage_start = time.time()

            self.exporter.export_excel(
                cluster_results,
                metadata,
                self.config['excel_output']
            )

            logger.info(f"Stage 7 completed in {time.time() - stage_start:.1f}s")

            # Create final summary
            summary = {
                'success': True,
                'processing_time': total_processing_time,
                'clusters_found': len(cluster_results),
                'points_original': point_data.point_count,
                'points_processed': len(points_array),
                'metadata': metadata,
                'cluster_results': cluster_results
            }

            # Log final results
            logger.info("\n" + "=" * 80)
            logger.info("PROCESSING COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total time: {format_duration(total_processing_time)}")
            logger.info(f"Points processed: {len(points_array):,} ({point_data.point_count:,} original)")
            logger.info(f"Clusters found: {len(cluster_results)}")
            logger.info(f"Results saved to: {self.config['excel_output']}")

            return summary

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"\nPIPELINE FAILED after {format_duration(error_time)}")
            logger.error(f"Error: {e}")

            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': error_time
            }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="E57 Point Cloud Processing Pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing with auto-selected clusters
  %(prog)s --input scan.e57 --excel_out results.xlsx

  # Specify number of clusters and preprocessing parameters
  %(prog)s --input scan.e57 --excel_out results.xlsx --k 5 --voxel 0.05

  # Use DBSCAN clustering with custom parameters
  %(prog)s --input scan.e57 --excel_out results.xlsx --method dbscan --eps 0.1 --min_samples 20

  # Export additional formats
  %(prog)s --input scan.e57 --excel_out results.xlsx --ply_out raw.ply --py7_out compressed.py7

  # Verbose output for debugging
  %(prog)s --input scan.e57 --excel_out results.xlsx --verbose

Notes:
  - Excel output is always in inches, regardless of input units
  - Use --k 0 for automatic cluster count selection (default)
  - Increase --voxel for faster processing of large files
  - Use --verbose for detailed logging output
        """
    )

    # Required arguments
    required = parser.add_argument_group('Required Arguments')
    required.add_argument(
        '--input',
        required=True,
        metavar='FILE',
        help='Input E57 point cloud file path'
    )
    required.add_argument(
        '--excel_out',
        required=True,
        metavar='FILE',
        help='Output Excel file path for results'
    )

    # Preprocessing parameters
    preprocessing = parser.add_argument_group('Preprocessing Parameters')
    preprocessing.add_argument(
        '--voxel',
        type=float,
        default=DEFAULT_VOXEL_SIZE,
        metavar='SIZE',
        help=f'Voxel size for downsampling in meters (default: {DEFAULT_VOXEL_SIZE})'
    )
    preprocessing.add_argument(
        '--nb_neighbors',
        type=int,
        default=DEFAULT_NB_NEIGHBORS,
        metavar='N',
        help=f'Number of neighbors for outlier removal (default: {DEFAULT_NB_NEIGHBORS})'
    )
    preprocessing.add_argument(
        '--std_ratio',
        type=float,
        default=DEFAULT_STD_RATIO,
        metavar='RATIO',
        help=f'Standard deviation ratio for outlier removal (default: {DEFAULT_STD_RATIO})'
    )

    # Clustering parameters
    clustering = parser.add_argument_group('Clustering Parameters')
    clustering.add_argument(
        '--method',
        choices=['kmeans', 'dbscan'],
        default='kmeans',
        help='Clustering algorithm to use (default: kmeans)'
    )
    clustering.add_argument(
        '--k',
        type=int,
        default=0,
        metavar='N',
        help='Number of clusters for K-means (0 = auto-select, default: 0)'
    )
    clustering.add_argument(
        '--eps',
        type=float,
        default=DEFAULT_DBSCAN_EPS,
        metavar='DISTANCE',
        help=f'DBSCAN epsilon parameter (default: {DEFAULT_DBSCAN_EPS})'
    )
    clustering.add_argument(
        '--min_samples',
        type=int,
        default=DEFAULT_DBSCAN_MIN_SAMPLES,
        metavar='N',
        help=f'DBSCAN minimum samples parameter (default: {DEFAULT_DBSCAN_MIN_SAMPLES})'
    )
    clustering.add_argument(
        '--min_cluster_points',
        type=int,
        default=DEFAULT_MIN_CLUSTER_POINTS,
        metavar='N',
        help=f'Minimum points required for valid cluster (default: {DEFAULT_MIN_CLUSTER_POINTS})'
    )

    # Export options
    export_group = parser.add_argument_group('Export Options')
    export_group.add_argument(
        '--ply_out',
        metavar='FILE',
        help='Export processed point cloud to PLY format'
    )
    export_group.add_argument(
        '--py7_out',
        metavar='FILE',
        help='Export raw data to compressed PY7 format'
    )

    # General options
    general = parser.add_argument_group('General Options')
    general.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    general.add_argument(
        '--version',
        action='version',
        version='E57 Point Cloud Processing Pipeline v2.0'
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid
        FileNotFoundError: If required files don't exist
    """
    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if not args.input.lower().endswith('.e57'):
        logger.warning(f"Input file '{args.input}' doesn't have .e57 extension")

    # Validate output directory
    output_dir = os.path.dirname(os.path.abspath(args.excel_out))
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    # Validate numeric parameters
    if args.voxel < 0:
        raise ValueError("Voxel size must be non-negative")

    if args.nb_neighbors <= 0:
        raise ValueError("Number of neighbors must be positive")

    if args.std_ratio <= 0:
        raise ValueError("Standard deviation ratio must be positive")

    if args.k < 0:
        raise ValueError("Number of clusters must be non-negative")

    if args.eps <= 0:
        raise ValueError("DBSCAN epsilon must be positive")

    if args.min_samples <= 0:
        raise ValueError("DBSCAN min_samples must be positive")

    if args.min_cluster_points <= 0:
        raise ValueError("Minimum cluster points must be positive")

    # Validate optional output files
    for output_file in [args.ply_out, args.py7_out]:
        if output_file:
            output_dir = os.path.dirname(os.path.abspath(output_file))
            if not os.path.exists(output_dir):
                raise FileNotFoundError(f"Output directory does not exist: {output_dir}")


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    try:
        # Validate arguments
        validate_arguments(args)

        # Check E57 library availability
        if not e57_deps.has_support:
            logger.error("No E57 library available!")
            logger.error("Please install one of the following:")
            logger.error("  pip install pye57")
            logger.error("  pip install e57")
            sys.exit(1)

        # Create pipeline configuration
        config = {
            # Input/Output
            'input_file': args.input,
            'excel_output': args.excel_out,
            'ply_output': args.ply_out,
            'py7_output': args.py7_out,

            # Preprocessing
            'voxel_size': args.voxel,
            'nb_neighbors': args.nb_neighbors,
            'std_ratio': args.std_ratio,

            # Clustering
            'cluster_method': args.method,
            'k': args.k,
            'eps': args.eps,
            'min_samples': args.min_samples,
            'min_cluster_points': args.min_cluster_points,
        }

        # Create and run pipeline
        pipeline = E57ProcessingPipeline(config)
        result = pipeline.run()

        # Handle results
        if result['success']:
            logger.info(f"\n🎉 Processing completed successfully!")
            logger.info(f"📊 Results saved to: {args.excel_out}")
            sys.exit(0)
        else:
            logger.error(f"\n❌ Processing failed: {result['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⏹️  Processing interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C

    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")

        if args.verbose:
            import traceback
            logger.error("Full traceback:")
            traceback.print_exc()

        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
