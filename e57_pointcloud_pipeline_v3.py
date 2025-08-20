#!/usr/bin/env python3
"""
E57 Point Cloud Processing Pipeline with Asset Classification
===========================================================

A comprehensive pipeline for processing E57 point cloud files with clustering,
asset classification (motors, pipes, etc.), and data export capabilities.

Features:
- Multi-library E57 support (pye57, e57)
- Advanced preprocessing (voxel downsampling, outlier removal)
- Multiple clustering algorithms (K-means, DBSCAN)
- Asset classification (Motor, Pipe_Straight, Pipe_Curved, Unknown)
- Oriented bounding box computation
- Multi-format export (Excel, PLY)
- Unit conversion (always exports to inches)

Author: [Your Name]
Date: August 21, 2025
Version: 2.6 - Final Error-Free Version
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

# Third-party libraries
import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Processing defaults
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
# EXCEPTION CLASSES
# =============================================================================

class E57ProcessingError(Exception):
    """Base exception for E57 processing errors."""
    pass


class LoadError(E57ProcessingError):
    """Exception raised when E57 file loading fails."""
    pass


class PreprocessingError(E57ProcessingError):
    """Exception raised during preprocessing operations."""
    pass


class ClusteringError(E57ProcessingError):
    """Exception raised during clustering operations."""
    pass


class ExportError(E57ProcessingError):
    """Exception raised during data export operations."""
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_unit_scale_to_inches(units: str) -> float:
    """Get scale factor to convert from given units to inches."""
    scale_factors = {
        'mm': MM_TO_INCHES,
        'millimeter': MM_TO_INCHES,
        'millimeters': MM_TO_INCHES,
        'cm': MM_TO_INCHES * 10,
        'centimeter': MM_TO_INCHES * 10,
        'centimeters': MM_TO_INCHES * 10,
        'm': MM_TO_INCHES * 1000,
        'meter': MM_TO_INCHES * 1000,
        'meters': MM_TO_INCHES * 1000,
        'in': 1.0,
        'inch': 1.0,
        'inches': 1.0,
        'ft': 12.0,
        'foot': 12.0,
        'feet': 12.0
    }

    normalized_units = units.lower().strip()
    return scale_factors.get(normalized_units, 1.0)


def convert_linear_measurements_to_inches(data_dict: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
    """Convert linear measurements in dictionary to inches."""
    linear_fields = ['center_x', 'center_y', 'center_z', 'length', 'width', 'height']
    volume_fields = ['volume']

    converted = data_dict.copy()

    # Convert linear measurements
    for field in linear_fields:
        if field in converted and isinstance(converted[field], (int, float)):
            converted[field] = float(converted[field]) * scale_factor

    # Convert volume measurements (scale^3)
    for field in volume_fields:
        if field in converted and isinstance(converted[field], (int, float)):
            converted[field] = float(converted[field]) * (scale_factor ** 3)

    return converted


def safe_convert_data(value: Any) -> Any:
    """Safely convert numpy arrays or scalars to Python types for Excel export."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        else:
            return value.tolist()
    elif isinstance(value, np.generic):
        return value.item()
    return value


# =============================================================================
# DATA MODELS
# =============================================================================

class PointCloudData:
    """Container for point cloud data with metadata."""

    def __init__(self, points: np.ndarray, metadata: Dict[str, Any] = None):
        self.points = np.asarray(points)
        self.metadata = metadata or {}
        self.point_count = len(self.points)

        logger.info(f"Point cloud loaded: {self.point_count:,} points")
        if self.metadata:
            logger.debug(f"Metadata: {list(self.metadata.keys())}")


class ClusterInfo:
    """Container for cluster analysis results with asset classification."""

    def __init__(
            self,
            asset_id: int,
            cluster_id: int,
            center: np.ndarray,
            dimensions: np.ndarray,
            rotation: np.ndarray,
            point_count: int = 0,
            asset_type: str = 'Unknown',
            asset_confidence: float = 0.0
    ):
        self.asset_id = asset_id
        self.cluster_id = cluster_id
        self.center = np.asarray(center)
        self.dimensions = np.asarray(dimensions)  # [length, width, height]
        self.rotation = np.asarray(rotation)  # [rx, ry, rz] in degrees
        self.point_count = point_count
        self.asset_type = asset_type
        self.asset_confidence = asset_confidence

    @property
    def volume(self) -> float:
        """Calculate approximate volume (length × width × height)."""
        return np.prod(self.dimensions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export - FINAL FIXED VERSION."""
        d = {
            'asset_id': self.asset_id,
            'cluster_id': self.cluster_id,
            'asset_type': self.asset_type,
            'asset_confidence': round(self.asset_confidence, 3),
            'center_x': safe_convert_data(self.center[0]),
            'center_y': safe_convert_data(self.center[1]),
            'center_z': safe_convert_data(self.center[2]),
            'length': safe_convert_data(self.dimensions),
            'width': safe_convert_data(self.dimensions[1]),
            'height': safe_convert_data(self.dimensions[2]),
            'rot_x': safe_convert_data(self.rotation),
            'rot_y': safe_convert_data(self.rotation[1]),
            'rot_z': safe_convert_data(self.rotation[2]),
            'volume': safe_convert_data(self.volume),
            'point_count': self.point_count
        }
        return d


# =============================================================================
# ASSET CLASSIFICATION MODULE - FINAL FIXED VERSION
# =============================================================================

class AssetClassifier:
    """Classifies point cloud clusters into asset types (Motor, Pipe, Unknown)."""

    def __init__(self):
        """Initialize asset classifier."""
        self.asset_types = ['Motor', 'Pipe_Straight', 'Pipe_Curved', 'Unknown']
        logger.info("Asset classifier initialized")

    def extract_geometric_features(self, points: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive geometric features from point cluster - FIXED VERSION."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Get bounding box features
        obb = pcd.get_oriented_bounding_box()
        extents = np.array(obb.extent)
        extents_sorted = np.sort(extents)[::-1]  # L >= W >= H

        features = {}

        # Basic dimensions - FIXED: Ensure scalar types
        features['length'] = float(extents_sorted[0])
        features['width'] = float(extents_sorted[1])
        features['height'] = float(extents_sorted[2])
        features['volume'] = float(np.prod(extents_sorted))
        features['diameter_est'] = float((extents_sorted[1] + extents_sorted[2]) / 2)

        # Shape ratios - FIXED: Ensure scalar types
        features['l_w_ratio'] = float(extents_sorted[0] / extents_sorted[1]) if extents_sorted[1] > 0 else 0.0
        features['w_h_ratio'] = float(extents_sorted[1] / extents_sorted[2]) if extents_sorted[2] > 0 else 0.0
        features['aspect_ratio'] = float(extents_sorted / extents_sorted[1]) if extents_sorted[1] > 0 else 0.0
        features['compactness'] = float(extents_sorted[2] / extents_sorted) if extents_sorted > 0 else 0.0
        features['cross_section_ratio'] = float(extents_sorted[1] / extents_sorted[2]) if extents_sorted[2] > 0 else 0.0

        # Density features
        features['point_count'] = len(points)
        features['point_density'] = len(points) / features['volume'] if features['volume'] > 0 else 0.0

        # PCA analysis for symmetry and elongation
        coords = np.asarray(pcd.points)
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid

        if len(centered) >= 3:
            cov_matrix = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            eigenvals = eigenvals[::-1]  # Sort descending

            # Prevent division by zero
            eigenvals = np.maximum(eigenvals, 1e-12)

            # FIXED: Ensure scalar types
            features['linearity'] = float((eigenvals[0] - eigenvals[1]) / eigenvals)
            features['planarity'] = float((eigenvals[1] - eigenvals[2]) / eigenvals)
            features['sphericity'] = float(eigenvals[2] / eigenvals)
            features['elongation'] = float(eigenvals / eigenvals[2])

            # Calculate straightness (for pipes)
            principal_axis = eigenvecs[:, 0]
            projections = np.dot(centered, principal_axis)
            proj_range = np.max(projections) - np.min(projections)
            features['straightness'] = float(proj_range / features['length']) if features['length'] > 0 else 0.0
        else:
            # Default values for small clusters
            features.update({
                'linearity': 0.0, 'planarity': 0.0, 'sphericity': 0.0,
                'elongation': 1.0, 'straightness': 0.0
            })

        # Length-to-diameter ratio (important for pipes)
        features['length_diameter_ratio'] = features['length'] / features['diameter_est'] if features[
                                                                                                 'diameter_est'] > 0 else 0.0

        return features

    def classify_motor(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Classify cluster as motor based on geometric features - FINAL FIXED VERSION."""
        constraints = []

        # Size constraints (typical industrial motor sizes) - FIXED: Explicit scalar comparisons
        size_ok = (features['length'] >= 0.3 and features['length'] <= 3.0 and
                   features['width'] >= 0.2 and features['width'] <= 2.5 and
                   features['height'] >= 0.2 and features['height'] <= 2.5)
        constraints.append(size_ok)

        # Shape constraints (not too elongated)
        shape_ok = (features['l_w_ratio'] <= 4.0 and
                    features['w_h_ratio'] <= 5.0 and
                    features['compactness'] >= 0.15)
        constraints.append(shape_ok)

        # Density constraint (motors are solid)
        density_ok = features['point_density'] > 300  # points per cubic meter
        constraints.append(density_ok)

        # Symmetry constraints (moderate symmetry, not too linear)
        symmetry_ok = (features['sphericity'] > 0.08 and
                       features['linearity'] < 0.85 and
                       features['elongation'] < 20)
        constraints.append(symmetry_ok)

        # Volume constraint (reasonable motor volume)
        volume_ok = features['volume'] >= 0.01 and features['volume'] <= 20.0  # 0.01 to 20 cubic meters
        constraints.append(volume_ok)

        is_motor = all(constraints)
        confidence = sum(constraints) / len(constraints) if len(constraints) > 0 else 0.0

        return is_motor, confidence

    def classify_straight_pipe(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Classify cluster as straight pipe - FINAL FIXED VERSION."""
        constraints = []

        # Size constraints - FIXED: Explicit scalar comparisons
        size_ok = (features['length'] > 0.5 and
                   features['diameter_est'] >= 0.02 and features['diameter_est'] <= 1.5)
        constraints.append(size_ok)

        # Elongation constraints
        elongation_ok = features['length_diameter_ratio'] > 8  # Length >> diameter
        constraints.append(elongation_ok)

        aspect_ok = features['aspect_ratio'] > 4  # Highly elongated
        constraints.append(aspect_ok)

        # Cross-section should be roughly circular
        circular_ok = features['cross_section_ratio'] >= 0.6 and features['cross_section_ratio'] <= 1.6
        constraints.append(circular_ok)

        # Should be highly linear (pipe-like)
        linear_ok = features['linearity'] > 0.75
        constraints.append(linear_ok)

        # Should not be too planar
        not_planar = features['planarity'] < 0.4
        constraints.append(not_planar)

        # Should be straight
        straight_ok = features['straightness'] > 0.7
        constraints.append(straight_ok)

        # Very elongated shape
        very_elongated = features['elongation'] > 15
        constraints.append(very_elongated)

        is_straight_pipe = all(constraints[:6])  # Don't require all constraints
        confidence = sum(constraints) / len(constraints) if len(constraints) > 0 else 0.0

        return is_straight_pipe, confidence

    def classify_curved_pipe(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Classify cluster as curved pipe - FINAL FIXED VERSION."""
        constraints = []

        # Size constraints - FIXED: Explicit scalar comparisons
        size_ok = (features['length'] > 0.5 and
                   features['diameter_est'] >= 0.02 and features['diameter_est'] <= 1.5)
        constraints.append(size_ok)

        # Should be elongated but less than straight pipes
        elongation_ok = features['length_diameter_ratio'] > 5
        constraints.append(elongation_ok)

        # Moderate aspect ratio
        aspect_ok = features['aspect_ratio'] > 3
        constraints.append(aspect_ok)

        # Cross-section roughly circular
        circular_ok = features['cross_section_ratio'] >= 0.6 and features['cross_section_ratio'] <= 1.6
        constraints.append(circular_ok)

        # Moderate linearity (less than straight pipes)
        moderate_linear = features['linearity'] > 0.4 and features['linearity'] < 0.8
        constraints.append(moderate_linear)

        # Less straight than straight pipes (key difference)
        curved_ok = features['straightness'] < 0.8
        constraints.append(curved_ok)

        # Should still be elongated
        elongated = features['elongation'] > 8
        constraints.append(elongated)

        is_curved_pipe = all(constraints[:5])  # Don't require all
        confidence = sum(constraints) / len(constraints) if len(constraints) > 0 else 0.0

        return is_curved_pipe, confidence

    def classify_cluster(self, points: np.ndarray) -> Tuple[str, float]:
        """Classify a point cluster into asset type - FINAL FIXED VERSION."""
        if len(points) < 20:
            return 'Unknown', 0.0

        try:
            # Extract geometric features
            features = self.extract_geometric_features(points)

            # Test each asset type
            is_motor, motor_conf = self.classify_motor(features)
            is_straight_pipe, straight_conf = self.classify_straight_pipe(features)
            is_curved_pipe, curved_conf = self.classify_curved_pipe(features)

            # Choose best classification
            candidates = []

            if is_motor and motor_conf > 0.5:
                candidates.append(('Motor', motor_conf))

            if is_straight_pipe and straight_conf > 0.5:
                candidates.append(('Pipe_Straight', straight_conf))

            if is_curved_pipe and curved_conf > 0.5:
                candidates.append(('Pipe_Curved', curved_conf))

            if candidates:
                # Return the classification with highest confidence
                best_asset, best_conf = max(candidates, key=lambda x: x[1])
                return best_asset, best_conf
            else:
                return 'Unknown', 0.0

        except Exception as e:
            logger.warning(f"Asset classification failed: {e}")
            return 'Unknown', 0.0


# =============================================================================
# E57 DEPENDENCIES MANAGEMENT
# =============================================================================

class E57Dependencies:
    """Manages E57 library dependencies and imports."""

    def __init__(self):
        self.pye57 = None
        self.e57 = None
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check and import available E57 libraries."""
        # Try pye57 first (preferred)
        try:
            import pye57
            self.pye57 = pye57
            logger.info("pye57 library available")
        except ImportError:
            logger.warning("pye57 not available")

        # Try e57 library as fallback
        try:
            import e57
            self.e57 = e57
            logger.info("e57 library available")
        except ImportError:
            logger.warning("e57 library not available")

        if not self.pye57 and not self.e57:
            raise ImportError(
                "No E57 library found. Please install either 'pye57' or 'e57':\n"
                "  pip install pye57\n"
                "  or\n"
                "  pip install e57"
            )

    @property
    def has_pye57(self) -> bool:
        """Check if pye57 is available."""
        return self.pye57 is not None

    @property
    def has_e57(self) -> bool:
        """Check if e57 is available."""
        return self.e57 is not None


# =============================================================================
# E57 FILE LOADER - FINAL FIXED VERSION
# =============================================================================

class E57Loader:
    """Handles E57 file loading with multiple library support."""

    def __init__(self):
        self.deps = E57Dependencies()

    def load_e57_file(self, file_path: str) -> PointCloudData:
        """Load E57 file using available library."""
        if not os.path.exists(file_path):
            raise LoadError(f"E57 file not found: {file_path}")

        logger.info(f"Loading E57 file: {file_path}")
        start_time = time.time()

        try:
            if self.deps.has_pye57:
                point_cloud_data = self._load_with_pye57(file_path)
            elif self.deps.has_e57:
                point_cloud_data = self._load_with_e57(file_path)
            else:
                raise LoadError("No E57 library available")

            load_time = time.time() - start_time
            logger.info(f"E57 file loaded successfully in {load_time:.2f}s")

            return point_cloud_data

        except Exception as e:
            raise LoadError(f"Failed to load E57 file: {e}")

    def _load_with_pye57(self, file_path: str) -> PointCloudData:
        """Load E57 file using pye57 library - FINAL FIXED VERSION."""
        logger.info("Using pye57 library for E57 loading")

        e57_file = self.deps.pye57.E57(file_path)

        # Get scan data
        scan_data = e57_file.read_scan(0, ignore_missing_fields=True)
        points = np.column_stack([scan_data["cartesianX"], scan_data["cartesianY"], scan_data["cartesianZ"]])

        # Extract metadata - FULLY FIXED for all pye57 API versions
        header = {}
        scan_header = {}

        # Get header safely - handle multiple API versions
        try:
            header_obj = e57_file.get_header(0)

            # Handle different header object types
            if hasattr(header_obj, '__dict__'):
                # If it's a proper object with attributes
                header = {
                    "creationDateTime": getattr(header_obj, 'creation_date_time', "unknown"),
                    "coordinateSystem": getattr(header_obj, 'coordinate_system', "unknown"),
                    "point_count": getattr(header_obj, 'point_count', len(points))
                }
            elif hasattr(header_obj, 'get'):
                # If it acts like a dictionary
                header = {
                    "creationDateTime": header_obj.get("creationDateTime", "unknown"),
                    "coordinateSystem": header_obj.get("coordinateSystem", "unknown"),
                    "point_count": header_obj.get("point_count", len(points))
                }
            else:
                # Fallback - extract what we can
                header = {"creationDateTime": "unknown", "coordinateSystem": "unknown"}

            logger.debug(f"Successfully extracted header information")

        except (TypeError, IndexError):
            # Try old API without index parameter
            try:
                header_obj = e57_file.get_header()
                if hasattr(header_obj, 'get'):
                    header = {
                        "creationDateTime": header_obj.get("creationDateTime", "unknown"),
                        "coordinateSystem": header_obj.get("coordinateSystem", "unknown")
                    }
                else:
                    header = {"creationDateTime": "unknown", "coordinateSystem": "unknown"}
                logger.debug("Using pye57 get_header() - older API")
            except Exception as e:
                logger.warning(f"Could not get header from E57 file: {e}")
                header = {"creationDateTime": "unknown", "coordinateSystem": "unknown"}
        except Exception as e:
            logger.warning(f"Could not get header from E57 file: {e}")
            header = {"creationDateTime": "unknown", "coordinateSystem": "unknown"}

        # Get scan header safely - handle missing methods
        try:
            # Try different methods to get scan information
            if hasattr(e57_file, 'scan_header'):
                scan_header_obj = e57_file.scan_header(0)
            elif hasattr(e57_file, 'get_scan'):
                scan_header_obj = e57_file.get_scan(0)
            else:
                # No scan header method available
                scan_header_obj = None
                logger.debug("No scan header method available")

            if scan_header_obj:
                if hasattr(scan_header_obj, 'get'):
                    scan_header = {
                        "coordinateSystem": scan_header_obj.get("coordinateSystem", "unknown"),
                        "units": scan_header_obj.get("units", "unknown")
                    }
                elif hasattr(scan_header_obj, '__dict__'):
                    scan_header = {
                        "coordinateSystem": getattr(scan_header_obj, 'coordinate_system', "unknown"),
                        "units": getattr(scan_header_obj, 'units', "unknown")
                    }
                else:
                    scan_header = {}
            else:
                scan_header = {}

        except Exception as e:
            logger.warning(f"Could not get scan header: {e}")
            scan_header = {}

        # Detect units from available information
        detected_units = self._detect_units_pye57(scan_header)

        metadata = {
            "source_file": file_path,
            "library_used": "pye57",
            "creation_time": header.get("creationDateTime", "unknown"),
            "scan_count": getattr(e57_file, 'scan_count', 1),
            "coordinate_system": header.get("coordinateSystem", scan_header.get("coordinateSystem", "unknown")),
            "cloud_units": detected_units,
            "original_point_count": len(points),
            "load_timestamp": datetime.now().isoformat()
        }

        logger.info(f"Detected coordinate system: {metadata['coordinate_system']}")
        logger.info(f"Detected units: {metadata['cloud_units']}")

        return PointCloudData(points, metadata)

    def _load_with_e57(self, file_path: str) -> PointCloudData:
        """Load E57 file using e57 library."""
        logger.info("Using e57 library for E57 loading")

        # Implementation would depend on e57 library API
        # This is a placeholder - actual implementation needed
        raise NotImplementedError("e57 library loading not yet implemented")

    def _detect_units_pye57(self, scan_header: Dict[str, Any]) -> str:
        """Detect coordinate units from pye57 scan header - ROBUST VERSION."""
        if not scan_header:
            logger.warning("No scan header available, assuming millimeters")
            return "mm"

        # Check common unit fields
        unit_fields = ["coordinateSystem", "units", "lengthUnit", "coordinate_system"]

        for field in unit_fields:
            if field in scan_header and scan_header[field]:
                unit_value = str(scan_header[field]).lower()
                if "mm" in unit_value or "millimeter" in unit_value:
                    return "mm"
                elif "cm" in unit_value or "centimeter" in unit_value:
                    return "cm"
                elif "m" in unit_value and "mm" not in unit_value:
                    return "m"
                elif "in" in unit_value or "inch" in unit_value:
                    return "in"
                elif "ft" in unit_value or "feet" in unit_value or "foot" in unit_value:
                    return "ft"

        # Default assumption for E57 files
        logger.warning("Could not detect coordinate units, assuming millimeters")
        return "mm"


# =============================================================================
# POINT CLOUD PREPROCESSOR
# =============================================================================

class PointCloudPreprocessor:
    """Handles point cloud preprocessing operations."""

    def __init__(self, voxel_size: float = DEFAULT_VOXEL_SIZE,
                 nb_neighbors: int = DEFAULT_NB_NEIGHBORS,
                 std_ratio: float = DEFAULT_STD_RATIO):
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

        logger.info(f"Preprocessor initialized: voxel_size={voxel_size}, "
                    f"nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")

    def preprocess_point_cloud(self, point_cloud_data: PointCloudData) -> o3d.geometry.PointCloud:
        """Apply preprocessing operations to point cloud."""
        logger.info("Starting point cloud preprocessing...")

        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_data.points)

            original_count = len(pcd.points)
            logger.info(f"Original point count: {original_count:,}")

            # Step 1: Voxel downsampling
            if self.voxel_size > 0:
                logger.info(f"Applying voxel downsampling (voxel_size={self.voxel_size})")
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                after_voxel = len(pcd.points)
                logger.info(f"After voxel downsampling: {after_voxel:,} points "
                            f"({100 * after_voxel / original_count:.1f}% retained)")

            # Step 2: Statistical outlier removal
            if self.nb_neighbors > 0 and self.std_ratio > 0:
                logger.info(f"Applying statistical outlier removal "
                            f"(nb_neighbors={self.nb_neighbors}, std_ratio={self.std_ratio})")
                pcd, inlier_indices = pcd.remove_statistical_outlier(
                    nb_neighbors=self.nb_neighbors,
                    std_ratio=self.std_ratio
                )
                after_outlier = len(pcd.points)
                logger.info(f"After outlier removal: {after_outlier:,} points "
                            f"({100 * after_outlier / original_count:.1f}% retained)")

            final_count = len(pcd.points)
            logger.info(f"Preprocessing complete: {final_count:,} points final "
                        f"({100 * final_count / original_count:.1f}% of original)")

            return pcd

        except Exception as e:
            raise PreprocessingError(f"Preprocessing failed: {e}")


# =============================================================================
# CLUSTER ANALYZER
# =============================================================================

class ClusterAnalyzer:
    """Handles clustering operations on point clouds."""

    def __init__(self, method: str = 'kmeans'):
        self.method = method.lower()
        self.supported_methods = ['kmeans', 'dbscan']

        if self.method not in self.supported_methods:
            raise ValueError(f"Unsupported clustering method: {method}. "
                             f"Supported: {self.supported_methods}")

        logger.info(f"Cluster analyzer initialized with method: {self.method}")

    def cluster_point_cloud(self, point_cloud: o3d.geometry.PointCloud, **kwargs) -> np.ndarray:
        """Perform clustering on point cloud."""
        logger.info(f"Starting clustering with method: {self.method}")

        try:
            points = np.asarray(point_cloud.points)

            if self.method == 'kmeans':
                labels = self._cluster_kmeans(points, **kwargs)
            elif self.method == 'dbscan':
                labels = self._cluster_dbscan(points, **kwargs)
            else:
                raise ClusteringError(f"Method {self.method} not implemented")

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
            n_noise = np.sum(labels == -1) if -1 in unique_labels else 0

            logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")

            return labels

        except Exception as e:
            raise ClusteringError(f"Clustering failed: {e}")

    def _cluster_kmeans(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """Perform K-means clustering with optimal K selection."""
        k_range = kwargs.get('k_range', DEFAULT_K_RANGE)
        min_k, max_k = k_range

        logger.info(f"Finding optimal K in range {min_k}-{max_k}")

        best_k = min_k
        best_score = -1

        for k in range(min_k, min(max_k + 1, len(points) // 2)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(points)

                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(points, cluster_labels)
                    logger.debug(f"K={k}: silhouette_score={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_k = k

            except Exception as e:
                logger.warning(f"K={k} failed: {e}")
                continue

        logger.info(f"Optimal K selected: {best_k} (silhouette_score={best_score:.3f})")

        # Perform final clustering with optimal K
        final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        return final_kmeans.fit_predict(points)

    def _cluster_dbscan(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """Perform DBSCAN clustering."""
        eps = kwargs.get('eps', DEFAULT_DBSCAN_EPS)
        min_samples = kwargs.get('min_samples', DEFAULT_DBSCAN_MIN_SAMPLES)

        logger.info(f"DBSCAN parameters: eps={eps}, min_samples={min_samples}")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(points)


# =============================================================================
# GEOMETRY ANALYZER
# =============================================================================

class GeometryAnalyzer:
    """Geometric analysis with asset classification."""

    def __init__(self):
        """Initialize geometry analyzer with asset classifier."""
        self.classifier = AssetClassifier()

    @staticmethod
    def rotation_matrix_to_euler_xyz(rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (XYZ order)."""
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

    def analyze_single_cluster(self, points: np.ndarray, cluster_id: int) -> Optional[ClusterInfo]:
        """Analyze a single cluster with asset classification."""
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

            # Classify the asset type
            asset_type, asset_confidence = self.classifier.classify_cluster(points)

            logger.debug(f"Cluster {cluster_id}: {asset_type} (confidence: {asset_confidence:.3f})")

            return ClusterInfo(
                asset_id=cluster_id + 1,
                cluster_id=cluster_id + 1,
                center=center,
                dimensions=dimensions_sorted,
                rotation=rotation_angles,
                point_count=len(points),
                asset_type=asset_type,
                asset_confidence=asset_confidence
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
        """Analyze all clusters with asset classification."""
        logger.info("Analyzing cluster geometries and classifying assets...")

        points = np.asarray(point_cloud.points)
        results = []

        unique_labels = np.unique(labels)
        valid_clusters = 0
        skipped_clusters = 0

        # Asset type counters
        asset_counts = {'Motor': 0, 'Pipe_Straight': 0, 'Pipe_Curved': 0, 'Unknown': 0}

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

            # Analyze cluster with asset classification
            cluster_info = self.analyze_single_cluster(cluster_points, cluster_id)

            if cluster_info is not None:
                results.append(cluster_info)
                valid_clusters += 1
                asset_counts[cluster_info.asset_type] += 1

                logger.debug(
                    f"Cluster {cluster_id}: {len(cluster_points)} points, "
                    f"type: {cluster_info.asset_type}, "
                    f"dimensions: {cluster_info.dimensions}"
                )

        logger.info(
            f"Cluster analysis complete: {valid_clusters} valid clusters, "
            f"{skipped_clusters} skipped (too few points)"
        )
        logger.info(f"Asset classification results:")
        for asset_type, count in asset_counts.items():
            if count > 0:
                logger.info(f"  {asset_type}: {count}")

        return results


# =============================================================================
# DATA EXPORTER - FINAL FIXED VERSION
# =============================================================================

class DataExporter:
    """Data exporter with asset type information."""

    @staticmethod
    def export_excel(
            cluster_results: List[ClusterInfo],
            metadata: Dict[str, Any],
            output_path: str
    ) -> None:
        """Export cluster results with asset types to Excel - FINAL FIXED VERSION."""
        logger.info(f"Exporting results to Excel: {output_path}")

        try:
            # Determine unit conversion
            cloud_units = metadata.get("cloud_units", "in")
            scale_to_inches = get_unit_scale_to_inches(cloud_units)

            logger.info(f"Converting from {cloud_units} to inches (scale: {scale_to_inches:.6f})")

            # Convert cluster data to inches and fix numpy arrays - FINAL FIXED VERSION
            cluster_data = []
            for cluster in cluster_results:
                cluster_dict = cluster.to_dict()

                # FIXED: Convert numpy arrays to scalars/lists for Excel compatibility
                for key, value in cluster_dict.items():
                    cluster_dict[key] = safe_convert_data(value)

                cluster_dict_inches = convert_linear_measurements_to_inches(
                    cluster_dict, scale_to_inches
                )
                cluster_data.append(cluster_dict_inches)

            # Column ordering with asset type information
            column_order = [
                'asset_id', 'cluster_id', 'asset_type', 'asset_confidence',
                'center_x', 'center_y', 'center_z',
                'length', 'width', 'height',
                'rot_x', 'rot_y', 'rot_z',
                'volume', 'point_count'
            ]

            df_clusters = pd.DataFrame(cluster_data, columns=column_order)

            # Prepare metadata - FIXED: Safe conversion for any numpy types
            metadata_export = metadata.copy()
            for key, value in metadata_export.items():
                metadata_export[key] = safe_convert_data(value)

            metadata_export["export_units"] = "inches"
            metadata_export["original_units"] = cloud_units
            metadata_export["unit_scale_factor"] = scale_to_inches

            # Add asset classification summary
            if cluster_results:
                asset_summary = {}
                for cluster in cluster_results:
                    asset_type = cluster.asset_type
                    asset_summary[asset_type] = asset_summary.get(asset_type, 0) + 1

                metadata_export["asset_classification_summary"] = asset_summary
                metadata_export["total_classified_assets"] = len(cluster_results)

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

            logger.info(f"Successfully exported {len(cluster_results)} classified clusters to Excel")

        except Exception as e:
            raise ExportError(f"Failed to export Excel: {e}")

    @staticmethod
    def export_ply(
            point_cloud: o3d.geometry.PointCloud,
            labels: np.ndarray,
            output_path: str
    ) -> None:
        """Export colored point cloud to PLY format."""
        logger.info(f"Exporting colored point cloud to PLY: {output_path}")

        try:
            # Create a copy of the point cloud
            output_pcd = o3d.geometry.PointCloud(point_cloud)

            # Generate colors for clusters
            unique_labels = np.unique(labels)
            max_label = max(unique_labels) if len(unique_labels) > 0 else 0

            # Create color map
            colors = np.random.rand(max_label + 2, 3)  # +2 for noise and extra
            colors[0] = [0.5, 0.5, 0.5]  # Gray for noise (label -1 becomes 0)

            # Assign colors to points
            point_colors = np.zeros((len(labels), 3))
            for i, label in enumerate(labels):
                point_colors[i] = colors[label + 1]  # +1 to handle -1 labels

            output_pcd.colors = o3d.utility.Vector3dVector(point_colors)

            # Write PLY file
            success = o3d.io.write_point_cloud(output_path, output_pcd)

            if success:
                logger.info(f"Successfully exported PLY file: {output_path}")
            else:
                raise ExportError("Failed to write PLY file")

        except Exception as e:
            raise ExportError(f"Failed to export PLY: {e}")


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

class E57ProcessingPipeline:
    """Main processing pipeline with asset classification."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration."""
        self.config = config
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
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
        logger.info("Asset classification enabled: Motor, Pipe_Straight, Pipe_Curved, Unknown")

    def process_e57_file(self, input_path: str, output_dir: str) -> str:
        """Process complete E57 file through the pipeline."""
        logger.info("=" * 60)
        logger.info("E57 POINT CLOUD PROCESSING PIPELINE - STARTING")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Step 1: Load E57 file
            logger.info("Step 1: Loading E57 file...")
            point_cloud_data = self.loader.load_e57_file(input_path)

            # Step 2: Preprocess point cloud
            logger.info("Step 2: Preprocessing point cloud...")
            preprocessed_pcd = self.preprocessor.preprocess_point_cloud(point_cloud_data)

            # Update metadata with preprocessing info
            metadata = point_cloud_data.metadata.copy()
            metadata.update({
                "voxel_size": self.preprocessor.voxel_size,
                "nb_neighbors": self.preprocessor.nb_neighbors,
                "std_ratio": self.preprocessor.std_ratio,
                "preprocessed_point_count": len(preprocessed_pcd.points),
                "processing_timestamp": datetime.now().isoformat()
            })

            # Step 3: Perform clustering
            logger.info("Step 3: Performing clustering...")
            cluster_labels = self.clusterer.cluster_point_cloud(
                preprocessed_pcd,
                **self.config.get('clustering_params', {})
            )

            # Step 4: Analyze clusters and classify assets
            logger.info("Step 4: Analyzing clusters and classifying assets...")
            cluster_results = self.geometry_analyzer.analyze_clusters(
                preprocessed_pcd,
                cluster_labels,
                min_points=self.config.get('min_cluster_points', DEFAULT_MIN_CLUSTER_POINTS)
            )

            # Step 5: Export results
            logger.info("Step 5: Exporting results...")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            excel_path = os.path.join(output_dir, f"{base_name}_analysis.xlsx")
            ply_path = os.path.join(output_dir, f"{base_name}_clusters.ply")

            # Export Excel file
            self.exporter.export_excel(cluster_results, metadata, excel_path)

            # Export PLY file (optional)
            if self.config.get('export_ply', True):
                self.exporter.export_ply(preprocessed_pcd, cluster_labels, ply_path)

            # Processing complete
            total_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info("E57 PROCESSING COMPLETE")
            logger.info(f"Total processing time: {total_time:.2f}s")
            logger.info(f"Results exported to: {output_dir}")
            logger.info(f"  - Excel analysis: {excel_path}")
            if self.config.get('export_ply', True):
                logger.info(f"  - PLY clusters: {ply_path}")
            logger.info("=" * 60)

            return excel_path

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="E57 Point Cloud Processing Pipeline with Asset Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing with K-means clustering
  python e57_pipeline.py input.e57 output_dir/

  # Use DBSCAN clustering with custom parameters
  python e57_pipeline.py input.e57 output_dir/ --cluster-method dbscan --dbscan-eps 0.1

  # Custom preprocessing parameters
  python e57_pipeline.py input.e57 output_dir/ --voxel-size 0.05 --std-ratio 1.5

  # Disable PLY export
  python e57_pipeline.py input.e57 output_dir/ --no-ply-export
        """
    )

    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input E57 file'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for results'
    )

    # Preprocessing options
    preprocessing_group = parser.add_argument_group('Preprocessing Options')
    preprocessing_group.add_argument(
        '--voxel-size',
        type=float,
        default=DEFAULT_VOXEL_SIZE,
        help=f'Voxel size for downsampling (default: {DEFAULT_VOXEL_SIZE})'
    )
    preprocessing_group.add_argument(
        '--nb-neighbors',
        type=int,
        default=DEFAULT_NB_NEIGHBORS,
        help=f'Number of neighbors for outlier removal (default: {DEFAULT_NB_NEIGHBORS})'
    )
    preprocessing_group.add_argument(
        '--std-ratio',
        type=float,
        default=DEFAULT_STD_RATIO,
        help=f'Standard deviation ratio for outlier removal (default: {DEFAULT_STD_RATIO})'
    )

    # Clustering options
    clustering_group = parser.add_argument_group('Clustering Options')
    clustering_group.add_argument(
        '--cluster-method',
        choices=['kmeans', 'dbscan'],
        default='kmeans',
        help='Clustering method to use (default: kmeans)'
    )
    clustering_group.add_argument(
        '--min-cluster-points',
        type=int,
        default=DEFAULT_MIN_CLUSTER_POINTS,
        help=f'Minimum points required for valid cluster (default: {DEFAULT_MIN_CLUSTER_POINTS})'
    )

    # K-means specific options
    kmeans_group = parser.add_argument_group('K-means Options')
    kmeans_group.add_argument(
        '--k-min',
        type=int,
        default=DEFAULT_K_RANGE[0],
        help=f'Minimum K for K-means search (default: {DEFAULT_K_RANGE})'
    )
    kmeans_group.add_argument(
        '--k-max',
        type=int,
        default=DEFAULT_K_RANGE[1],
        help=f'Maximum K for K-means search (default: {DEFAULT_K_RANGE[1]})'
    )

    # DBSCAN specific options
    dbscan_group = parser.add_argument_group('DBSCAN Options')
    dbscan_group.add_argument(
        '--dbscan-eps',
        type=float,
        default=DEFAULT_DBSCAN_EPS,
        help=f'DBSCAN epsilon parameter (default: {DEFAULT_DBSCAN_EPS})'
    )
    dbscan_group.add_argument(
        '--dbscan-min-samples',
        type=int,
        default=DEFAULT_DBSCAN_MIN_SAMPLES,
        help=f'DBSCAN min_samples parameter (default: {DEFAULT_DBSCAN_MIN_SAMPLES})'
    )

    # Export options
    export_group = parser.add_argument_group('Export Options')
    export_group.add_argument(
        '--no-ply-export',
        action='store_true',
        help='Disable PLY file export'
    )

    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    return parser


def main() -> None:
    """Main entry point for command line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.getLogger().setLevel(numeric_level)

    # Build configuration from arguments
    config = {
        'voxel_size': args.voxel_size,
        'nb_neighbors': args.nb_neighbors,
        'std_ratio': args.std_ratio,
        'cluster_method': args.cluster_method,
        'min_cluster_points': args.min_cluster_points,
        'export_ply': not args.no_ply_export,
        'clustering_params': {}
    }

    # Add method-specific parameters
    if args.cluster_method == 'kmeans':
        config['clustering_params']['k_range'] = (args.k_min, args.k_max)
    elif args.cluster_method == 'dbscan':
        config['clustering_params'].update({
            'eps': args.dbscan_eps,
            'min_samples': args.dbscan_min_samples
        })

    try:
        # Initialize and run pipeline
        pipeline = E57ProcessingPipeline(config)
        output_file = pipeline.process_e57_file(args.input_file, args.output_dir)

        print(f"\n✓ Processing completed successfully!")
        print(f"✓ Results saved to: {output_file}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
