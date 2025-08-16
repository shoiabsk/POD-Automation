"""
E57 Point Cloud Processing Pipeline
==================================

A comprehensive pipeline for processing E57 point cloud files with clustering,
bounding box computation, and data export capabilities.

Author: [Your Name]
Date: [Current Date]
Version: 2.0
"""

import argparse
import os
import sys
import json
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

class E57Dependencies:
    """Manages E57 library dependencies"""

    def __init__(self):
        self.pye57 = None
        self.e57 = None
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import pye57
            self.pye57 = pye57
            logger.info("Using pye57 library for E57 support")
        except ImportError:
            logger.warning("pye57 not available")

        if not self.pye57:
            try:
                import e57
                self.e57 = e57
                logger.info("Using e57 library for E57 support")
            except ImportError:
                logger.warning("e57 not available")

    def has_support(self) -> bool:
        return self.pye57 is not None or self.e57 is not None


# Global dependency manager
e57_deps = E57Dependencies()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class PointCloudData:
    """Container for point cloud data"""

    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                 intensity: Optional[np.ndarray] = None):
        self.points = points
        self.colors = colors
        self.intensity = intensity
        self.validate()

    def validate(self):
        """Validate data consistency"""
        if self.points is None or len(self.points) == 0:
            raise ValueError("Points array cannot be empty")

        n_points = len(self.points)
        if self.colors is not None and len(self.colors) != n_points:
            raise ValueError(f"Color array length ({len(self.colors)}) doesn't match points ({n_points})")

        if self.intensity is not None and len(self.intensity) != n_points:
            raise ValueError(f"Intensity array length ({len(self.intensity)}) doesn't match points ({n_points})")

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary format"""
        result = {"points": self.points}
        if self.colors is not None:
            result["colors"] = self.colors
        if self.intensity is not None:
            result["intensity"] = self.intensity
        return result


class ClusterInfo:
    """Container for cluster information"""

    def __init__(self, asset_id: int, cluster_id: int, center: np.ndarray,
                 dimensions: np.ndarray, rotation: np.ndarray):
        self.asset_id = asset_id
        self.cluster_id = cluster_id
        self.center = center
        self.dimensions = dimensions  # [length, width, height]
        self.rotation = rotation  # [rx, ry, rz] in degrees

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for export"""
        return {
            "asset_id": self.asset_id,
            "cluster_id": self.cluster_id,
            "center_x": self.center[0],
            "center_y": self.center[1],
            "center_z": self.center[2],
            "length": self.dimensions[0],
            "width": self.dimensions[1],
            "height": self.dimensions[2],
            "rot_x": self.rotation[0],
            "rot_y": self.rotation[1],
            "rot_z": self.rotation[2]
        }


# ============================================================================
# E57 FILE LOADING
# ============================================================================

class E57Loader:
    """Handles loading of E57 files with multiple library backends"""

    def __init__(self):
        if not e57_deps.has_support():
            raise ImportError("No E57 library available. Install 'pye57' or 'e57'")

    def load(self, file_path: str, want_color: bool = True,
             want_intensity: bool = True) -> PointCloudData:
        """Load point cloud data from E57 file"""

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"E57 file not found: {file_path}")

        logger.info(f"Loading E57 file: {file_path}")

        try:
            if e57_deps.pye57:
                return self._load_with_pye57(file_path, want_color, want_intensity)
            elif e57_deps.e57:
                return self._load_with_e57(file_path, want_color, want_intensity)
        except Exception as e:
            logger.error(f"Failed to load E57 file: {e}")
            raise

    def _load_with_pye57(self, file_path: str, want_color: bool,
                         want_intensity: bool) -> PointCloudData:
        """Load using pye57 library"""

        e = e57_deps.pye57.E57(file_path)
        logger.info(f"Found {e.scan_count} scans in E57 file")

        pts_list, cols_list, ints_list = [], [], []

        for scan_idx in range(e.scan_count):
            logger.info(f"Processing scan {scan_idx + 1}/{e.scan_count}")

            try:
                data = e.read_scan(scan_idx, intensity=want_intensity, colors=want_color)

                # Extract coordinates
                x, y, z = data.get("cartesianX"), data.get("cartesianY"), data.get("cartesianZ")
                if x is None or y is None or z is None:
                    logger.warning(f"Scan {scan_idx} missing coordinate data, skipping")
                    continue

                # Validate coordinate data
                if len(x) == 0 or len(y) == 0 or len(z) == 0:
                    logger.warning(f"Scan {scan_idx} has empty coordinate arrays, skipping")
                    continue

                pts_list.append(np.column_stack([x, y, z]))

                # Extract colors if available
                if want_color and all(k in data for k in ("colorRed", "colorGreen", "colorBlue")):
                    cr = np.asarray(data["colorRed"]) / 255.0
                    cg = np.asarray(data["colorGreen"]) / 255.0
                    cb = np.asarray(data["colorBlue"]) / 255.0
                    cols_list.append(np.column_stack([cr, cg, cb]))

                # Extract intensity if available
                if want_intensity and "intensity" in data and data["intensity"] is not None:
                    ints_list.append(np.asarray(data["intensity"]))

            except Exception as e:
                logger.warning(f"Error processing scan {scan_idx}: {e}")
                continue

        if not pts_list:
            raise ValueError("No valid point data found in E57 file")

        # Combine all scans
        points = np.vstack(pts_list)
        colors = np.vstack(cols_list) if cols_list else None
        intensity = np.concatenate(ints_list) if ints_list else None

        logger.info(f"Loaded {len(points)} points from {len(pts_list)} scans")
        return PointCloudData(points, colors, intensity)

    def _load_with_e57(self, file_path: str, want_color: bool,
                       want_intensity: bool) -> PointCloudData:
        """Load using e57 library"""

        pc = e57_deps.e57.read_points(file_path)

        points = np.asarray(pc.points)
        colors = np.asarray(pc.color) if hasattr(pc, "color") and pc.color is not None else None
        intensity = np.asarray(pc.intensity) if hasattr(pc, "intensity") and pc.intensity is not None else None

        logger.info(f"Loaded {len(points)} points")
        return PointCloudData(points, colors, intensity)


# ============================================================================
# POINT CLOUD PREPROCESSING
# ============================================================================

class PointCloudPreprocessor:
    """Handles point cloud preprocessing operations"""

    def __init__(self, voxel_size: float = 0.02, nb_neighbors: int = 20,
                 std_ratio: float = 2.0):
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def process(self, pcd_data: PointCloudData) -> o3d.geometry.PointCloud:
        """Preprocess point cloud data"""

        logger.info("Starting point cloud preprocessing...")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data.points)

        if pcd_data.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pcd_data.colors)

        original_count = len(pcd.points)
        logger.info(f"Original point count: {original_count}")

        # Voxel downsampling
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.voxel_size)
            logger.info(f"After voxel downsampling: {len(pcd.points)} points")

        # Statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio
        )

        final_count = len(pcd.points)
        logger.info(f"After outlier removal: {final_count} points")
        logger.info(
            f"Removed {original_count - final_count} points ({100 * (original_count - final_count) / original_count:.1f}%)")

        return pcd


# ============================================================================
# CLUSTERING
# ============================================================================

class ClusterAnalyzer:
    """Handles point cloud clustering operations"""

    def __init__(self, method: str = "kmeans"):
        self.method = method.lower()
        if self.method not in ["kmeans", "dbscan"]:
            raise ValueError("Method must be 'kmeans' or 'dbscan'")

    def auto_select_k(self, points: np.ndarray, k_min: int = 2, k_max: int = 12) -> int:
        """Automatically select optimal number of clusters using silhouette analysis"""

        logger.info(f"Selecting optimal k between {k_min} and {k_max}...")

        best_k, best_score = k_min, -1
        scores = {}

        for k in range(k_min, min(k_max + 1, len(points))):
            try:
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = kmeans.fit_predict(points)

                if len(set(labels)) < 2:
                    continue

                score = silhouette_score(points, labels)
                scores[k] = score

                logger.info(f"k={k}: silhouette score = {score:.3f}")

                if score > best_score:
                    best_k, best_score = k, score

            except Exception as e:
                logger.warning(f"Failed to evaluate k={k}: {e}")
                continue

        logger.info(f"Selected k={best_k} with silhouette score {best_score:.3f}")
        return best_k

    def cluster_kmeans(self, points: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Perform K-means clustering"""

        if k is None or k <= 0:
            k = self.auto_select_k(points)

        logger.info(f"Performing K-means clustering with k={k}")

        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(points)

        unique_labels = len(set(labels))
        logger.info(f"Created {unique_labels} clusters")

        return labels

    def cluster_dbscan(self, points: np.ndarray, eps: float = 0.05,
                       min_samples: int = 10) -> np.ndarray:
        """Perform DBSCAN clustering"""

        logger.info(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)

        unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = np.sum(labels == -1)

        logger.info(f"Created {unique_labels} clusters with {noise_points} noise points")

        return labels

    def cluster(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """Perform clustering based on selected method"""

        if self.method == "kmeans":
            return self.cluster_kmeans(points, kwargs.get("k"))
        elif self.method == "dbscan":
            return self.cluster_dbscan(points, kwargs.get("eps", 0.05), kwargs.get("min_samples", 10))


# ============================================================================
# GEOMETRY ANALYSIS
# ============================================================================

class GeometryAnalyzer:
    """Handles geometric analysis of clusters"""

    @staticmethod
    def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (XYZ order)"""

        # Clamp to avoid numerical issues
        r20 = max(min(R[2, 0], 1), -1)
        ry = math.asin(-r20)
        cy = math.cos(ry)

        if abs(cy) > 1e-8:
            rx = math.atan2(R[2, 1] / cy, R[2, 2] / cy)
            rz = math.atan2(R[1, 0] / cy, R[0, 0] / cy)
        else:
            rx = 0
            rz = math.atan2(-R[0, 1], R[1, 1])

        return np.degrees([rx, ry, rz])

    def analyze_cluster(self, points: np.ndarray, cluster_id: int) -> Optional[ClusterInfo]:
        """Analyze a single cluster and compute bounding box info"""

        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Compute AABB as fallback
            aabb = pcd.get_axis_aligned_bounding_box()
            center_fallback = aabb.get_center()
            extent_fallback = np.array(aabb.get_extent())

            # Try to compute OBB
            try:
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
                center = obb.get_center()
                extent = np.array(obb.extent)
                rotation_matrix = np.asarray(obb.R)
                rotation_angles = self.rotation_matrix_to_euler_xyz(rotation_matrix)
            except Exception as e:
                logger.warning(f"OBB computation failed for cluster {cluster_id}: {e}. Using AABB.")
                center = center_fallback
                extent = extent_fallback
                rotation_angles = np.array([0, 0, 0])

            # Sort dimensions: length >= width >= height
            sorted_extent = np.sort(extent)[::-1]

            return ClusterInfo(
                asset_id=cluster_id + 1,
                cluster_id=cluster_id + 1,
                center=center,
                dimensions=sorted_extent,
                rotation=rotation_angles
            )

        except Exception as e:
            logger.error(f"Failed to analyze cluster {cluster_id}: {e}")
            return None

    def analyze_clusters(self, pcd: o3d.geometry.PointCloud, labels: np.ndarray,
                         min_points: int = 200) -> List[ClusterInfo]:
        """Analyze all clusters and compute their properties"""

        logger.info("Analyzing cluster geometries...")

        points = np.asarray(pcd.points)
        results = []

        unique_labels = np.unique(labels)
        valid_clusters = 0

        for cluster_id in unique_labels:
            if cluster_id < 0:  # Skip noise points
                continue

            # Get cluster points
            cluster_mask = labels == cluster_id
            cluster_points = points[cluster_mask]

            if len(cluster_points) < min_points:
                logger.warning(f"Cluster {cluster_id} has only {len(cluster_points)} points (< {min_points}), skipping")
                continue

            cluster_info = self.analyze_cluster(cluster_points, cluster_id)
            if cluster_info:
                results.append(cluster_info)
                valid_clusters += 1

        logger.info(f"Successfully analyzed {valid_clusters} clusters")
        return results


# ============================================================================
# DATA EXPORT
# ============================================================================

class DataExporter:
    """Handles various data export formats"""

    @staticmethod
    def export_excel(cluster_results: List[ClusterInfo], metadata: Dict[str, Any],
                     output_path: str):
        """Export cluster results and metadata to Excel"""

        logger.info(f"Exporting results to Excel: {output_path}")

        # Convert cluster info to DataFrame
        cluster_data = [cluster.to_dict() for cluster in cluster_results]
        df_clusters = pd.DataFrame(cluster_data)

        # Create metadata DataFrame
        df_metadata = pd.DataFrame([metadata])

        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_clusters.to_excel(writer, sheet_name='clusters', index=False)
            df_metadata.to_excel(writer, sheet_name='metadata', index=False)

        logger.info(f"Exported {len(cluster_results)} clusters to Excel")

    @staticmethod
    def export_ply(pcd_data: PointCloudData, output_path: str):
        """Export point cloud to PLY format"""

        logger.info(f"Exporting point cloud to PLY: {output_path}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data.points)

        if pcd_data.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(pcd_data.colors)

        success = o3d.io.write_point_cloud(output_path, pcd)
        if not success:
            raise RuntimeError(f"Failed to write PLY file: {output_path}")

        logger.info(f"Successfully exported {len(pcd_data.points)} points to PLY")

    @staticmethod
    def export_py7(pcd_data: PointCloudData, output_path: str, source_filename: str):
        """Export to custom PY7 format (compressed numpy)"""

        logger.info(f"Exporting to PY7 format: {output_path}")

        metadata = {
            "source": source_filename,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "point_count": len(pcd_data.points),
            "has_colors": pcd_data.colors is not None,
            "has_intensity": pcd_data.intensity is not None
        }

        np.savez_compressed(
            output_path,
            points=pcd_data.points,
            colors=pcd_data.colors,
            intensity=pcd_data.intensity,
            metadata=json.dumps(metadata)
        )

        logger.info(f"Successfully exported to PY7 format")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class E57ProcessingPipeline:
    """Main processing pipeline orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loader = E57Loader()
        self.preprocessor = PointCloudPreprocessor(
            voxel_size=config.get('voxel_size', 0.02),
            nb_neighbors=config.get('nb_neighbors', 20),
            std_ratio=config.get('std_ratio', 2.0)
        )
        self.clusterer = ClusterAnalyzer(method=config.get('cluster_method', 'kmeans'))
        self.geometry_analyzer = GeometryAnalyzer()
        self.exporter = DataExporter()

    def run(self) -> Dict[str, Any]:
        """Execute the complete processing pipeline"""

        start_time = time.time()
        logger.info("Starting E57 processing pipeline...")

        try:
            # Step 1: Load E57 file
            pcd_data = self.loader.load(self.config['input_file'])

            # Step 2: Optional exports of raw data
            if self.config.get('ply_output'):
                self.exporter.export_ply(pcd_data, self.config['ply_output'])

            if self.config.get('py7_output'):
                self.exporter.export_py7(
                    pcd_data,
                    self.config['py7_output'],
                    os.path.basename(self.config['input_file'])
                )

            # Step 3: Preprocess point cloud
            processed_pcd = self.preprocessor.process(pcd_data)

            # Step 4: Perform clustering
            points = np.asarray(processed_pcd.points)
            cluster_params = {
                'k': self.config.get('k', 0),
                'eps': self.config.get('eps', 0.05),
                'min_samples': self.config.get('min_samples', 10)
            }
            labels = self.clusterer.cluster(points, **cluster_params)

            # Step 5: Analyze clusters
            cluster_results = self.geometry_analyzer.analyze_clusters(
                processed_pcd,
                labels,
                min_points=self.config.get('min_cluster_points', 200)
            )

            # Step 6: Create metadata
            metadata = {
                'input_file': os.path.basename(self.config['input_file']),
                'processing_time': time.time() - start_time,
                'total_points_original': len(pcd_data.points),
                'total_points_processed': len(points),
                'clusters_found': len(cluster_results),
                'voxel_size': self.config.get('voxel_size', 0.02),
                'nb_neighbors': self.config.get('nb_neighbors', 20),
                'std_ratio': self.config.get('std_ratio', 2.0),
                'cluster_method': self.config.get('cluster_method', 'kmeans'),
                'min_cluster_points': self.config.get('min_cluster_points', 200)
            }

            # Step 7: Export results
            self.exporter.export_excel(
                cluster_results,
                metadata,
                self.config['excel_output']
            )

            # Prepare summary
            summary = {
                'success': True,
                'processing_time': metadata['processing_time'],
                'clusters_found': len(cluster_results),
                'points_processed': len(points),
                'metadata': metadata
            }

            logger.info(f"Pipeline completed successfully in {summary['processing_time']:.1f}s")
            return summary

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""

    parser = argparse.ArgumentParser(
        description="E57 Point Cloud Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input scan.e57 --excel_out results.xlsx
  %(prog)s --input scan.e57 --excel_out results.xlsx --k 5 --voxel 0.05
  %(prog)s --input scan.e57 --excel_out results.xlsx --method dbscan --eps 0.1
        """
    )

    # Required arguments
    parser.add_argument(
        '--input',
        required=True,
        help='Input E57 point cloud file'
    )
    parser.add_argument(
        '--excel_out',
        required=True,
        help='Output Excel file for results'
    )

    # Preprocessing parameters
    preprocessing = parser.add_argument_group('Preprocessing')
    preprocessing.add_argument(
        '--voxel',
        type=float,
        default=0.02,
        help='Voxel size for downsampling (meters, default: 0.02)'
    )
    preprocessing.add_argument(
        '--nb_neighbors',
        type=int,
        default=20,
        help='Number of neighbors for outlier removal (default: 20)'
    )
    preprocessing.add_argument(
        '--std_ratio',
        type=float,
        default=2.0,
        help='Standard deviation ratio for outlier removal (default: 2.0)'
    )

    # Clustering parameters
    clustering = parser.add_argument_group('Clustering')
    clustering.add_argument(
        '--method',
        choices=['kmeans', 'dbscan'],
        default='kmeans',
        help='Clustering method (default: kmeans)'
    )
    clustering.add_argument(
        '--k',
        type=int,
        default=0,
        help='Number of clusters for K-means (0 = auto-select, default: 0)'
    )
    clustering.add_argument(
        '--eps',
        type=float,
        default=0.05,
        help='DBSCAN epsilon parameter (default: 0.05)'
    )
    clustering.add_argument(
        '--min_samples',
        type=int,
        default=10,
        help='DBSCAN minimum samples parameter (default: 10)'
    )
    clustering.add_argument(
        '--min_cluster_points',
        type=int,
        default=200,
        help='Minimum points required for a valid cluster (default: 200)'
    )

    # Export options
    export = parser.add_argument_group('Export Options')
    export.add_argument(
        '--ply_out',
        help='Optional export of processed point cloud to PLY format'
    )
    export.add_argument(
        '--py7_out',
        help='Optional export to custom PY7 format'
    )

    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser


def validate_arguments(args) -> None:
    """Validate command line arguments"""

    # Check input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if not args.input.lower().endswith('.e57'):
        logger.warning("Input file doesn't have .e57 extension")

    # Check output directory
    output_dir = os.path.dirname(args.excel_out)
    if output_dir and not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Validate parameters
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


def main():
    """Main entry point"""

    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate arguments
        validate_arguments(args)

        # Check E57 library availability
        if not e57_deps.has_support():
            logger.error("No E57 library available. Please install 'pye57' or 'e57':")
            logger.error("  pip install pye57")
            logger.error("  # or")
            logger.error("  pip install e57")
            sys.exit(1)

        # Create configuration
        config = {
            'input_file': args.input,
            'excel_output': args.excel_out,
            'ply_output': args.ply_out,
            'py7_output': args.py7_out,
            'voxel_size': args.voxel,
            'nb_neighbors': args.nb_neighbors,
            'std_ratio': args.std_ratio,
            'cluster_method': args.method,
            'k': args.k,
            'eps': args.eps,
            'min_samples': args.min_samples,
            'min_cluster_points': args.min_cluster_points
        }

        # Run pipeline
        pipeline = E57ProcessingPipeline(config)
        result = pipeline.run()

        if result['success']:
            logger.info("=" * 60)
            logger.info("PROCESSING COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Processing time: {result['processing_time']:.1f} seconds")
            logger.info(f"Clusters found: {result['clusters_found']}")
            logger.info(f"Points processed: {result['points_processed']}")
            logger.info(f"Results saved to: {args.excel_out}")
            sys.exit(0)
        else:
            logger.error(f"Processing failed: {result['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
