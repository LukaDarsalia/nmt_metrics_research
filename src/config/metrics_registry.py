"""
Central metrics registry for NMT evaluation.

This module provides a centralized way to register, discover, and manage
all available metrics, avoiding hardcoded dictionaries across multiple files.
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Type, Optional, Any
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')


@dataclass
class MetricInfo:
    """Information about a registered metric."""
    name: str
    class_name: str
    module_path: str
    script_path: str
    dependencies: List[str]
    supports_reference_free: bool = False
    supports_reference_based: bool = True
    description: str = ""


class MetricsRegistry:
    """
    Central registry for all available NMT metrics.

    This class automatically discovers and registers metrics, providing
    a single source of truth for available metrics and their properties.
    """

    def __init__(self):
        """Initialize the metrics registry."""
        self._metrics: Dict[str, MetricInfo] = {}
        self._initialized = False

    def register_metric(
        self,
        name: str,
        class_name: str,
        module_path: str,
        script_path: str,
        dependencies: List[str],
        supports_reference_free: bool = False,
        supports_reference_based: bool = True,
        description: str = ""
    ) -> None:
        """
        Register a metric in the registry.

        Args:
            name (str): Display name of the metric (e.g., 'BLEU', 'CHRF++')
            class_name (str): Name of the metric class (e.g., 'BLEUMetric')
            module_path (str): Import path to the module (e.g., 'src.metrics.bleu_metric')
            script_path (str): Path to the standalone script (e.g., 'src/metrics/bleu_metric.py')
            dependencies (List[str]): List of required packages
            supports_reference_free (bool): Whether metric supports reference-free evaluation
            supports_reference_based (bool): Whether metric supports reference-based evaluation
            description (str): Description of the metric
        """
        self._metrics[name] = MetricInfo(
            name=name,
            class_name=class_name,
            module_path=module_path,
            script_path=script_path,
            dependencies=dependencies,
            supports_reference_free=supports_reference_free,
            supports_reference_based=supports_reference_based,
            description=description
        )

    def auto_discover_metrics(self, metrics_dir: str = "src/metrics") -> None:
        """
        Automatically discover and register metrics from the metrics directory.

        Args:
            metrics_dir (str): Path to the metrics directory
        """
        if not os.path.exists(metrics_dir):
            print(f"⚠️  Metrics directory not found: {metrics_dir}")
            return

        print(f"🔍 Auto-discovering metrics in: {metrics_dir}")

        # Get all Python files in metrics directory
        metrics_path = Path(metrics_dir)
        python_files = list(metrics_path.glob("*_metric.py"))

        for file_path in python_files:
            try:
                self._discover_metric_from_file(file_path)
            except Exception as e:
                print(f"⚠️  Failed to discover metric from {file_path}: {e}")

    def _discover_metric_from_file(self, file_path: Path) -> None:
        """
        Discover and register a metric from a Python file.

        Args:
            file_path (Path): Path to the metric file
        """
        # Convert file path to module path
        module_name = file_path.stem
        module_path = f"src.metrics.{module_name}"

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find metric classes that inherit from BaseMetric
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (name.endswith('Metric') and
                    name != 'BaseMetric' and
                    hasattr(obj, 'calculate_scores')):

                    # Try to instantiate to get metric info
                    try:
                        if 'LLM' in name:
                            # Handle LLM metrics specially - they need mode parameter
                            continue  # Skip auto-discovery for LLM metrics, register manually
                        else:
                            instance = obj()
                            metric_name = instance.metric_name
                            dependencies = instance.get_dependencies()

                            self.register_metric(
                                name=metric_name,
                                class_name=name,
                                module_path=module_path,
                                script_path=str(file_path),
                                dependencies=dependencies,
                                description=f"Auto-discovered {metric_name} metric"
                            )

                            print(f"✅ Discovered metric: {metric_name}")

                    except Exception as e:
                        print(f"⚠️  Failed to instantiate {name}: {e}")

        except Exception as e:
            print(f"⚠️  Failed to import {file_path}: {e}")

    def _register_builtin_metrics(self) -> None:
        """Register built-in metrics with their specific configurations."""

        # Traditional metrics
        builtin_metrics = [
            {
                'name': 'BLEU',
                'class_name': 'BLEUMetric',
                'module_path': 'src.metrics.bleu_metric',
                'script_path': 'src.metrics.bleu_metric',
                'dependencies': ['sacrebleu'],
                'description': 'BLEU score for machine translation evaluation'
            },
            {
                'name': 'CHRF++',
                'class_name': 'CHRFMetric',
                'module_path': 'src.metrics.chrf_metric',
                'script_path': 'src.metrics.chrf_metric',
                'dependencies': ['sacrebleu'],
                'description': 'CHRF++ score, particularly good for morphologically rich languages'
            },
            {
                'name': 'TER',
                'class_name': 'TERMetric',
                'module_path': 'src.metrics.ter_metric',
                'script_path': 'src.metrics.ter_metric',
                'dependencies': ['evaluate', 'datasets'],
                'description': 'Translation Edit Rate metric'
            },
            {
                'name': 'ROUGE',
                'class_name': 'ROUGEMetric',
                'module_path': 'src.metrics.rouge_metric',
                'script_path': 'src.metrics.rouge_metric',
                'dependencies': ['rouge_score'],
                'description': 'ROUGE-L score for translation evaluation'
            },
            {
                'name': 'COMET',
                'class_name': 'COMETMetric',
                'module_path': 'src.metrics.comet_metric',
                'script_path': 'src.metrics.comet_metric',
                'dependencies': ['unbabel-comet', 'torch', 'transformers'],
                'description': 'Neural-based COMET metric using multilingual models'
            }
        ]

        for metric_info in builtin_metrics:
            self.register_metric(**metric_info)

        # LLM metrics (special handling)
        llm_metrics = [
            {
                'name': 'LLM-Reference-Based',
                'class_name': 'LLMMetric',
                'module_path': 'src.metrics.llm_metric',
                'script_path': 'src.metrics.llm_metric',
                'dependencies': ['anthropic', 'python-dotenv', 'requests'],
                'supports_reference_free': False,
                'supports_reference_based': True,
                'description': 'LLM-based evaluation using reference translations'
            },
            {
                'name': 'LLM-Reference-Free',
                'class_name': 'LLMMetric',
                'module_path': 'src.metrics.llm_metric',
                'script_path': 'src.metrics.llm_metric',
                'dependencies': ['anthropic', 'python-dotenv', 'requests'],
                'supports_reference_free': True,
                'supports_reference_based': False,
                'description': 'LLM-based evaluation without reference translations'
            }
        ]

        for metric_info in llm_metrics:
            self.register_metric(**metric_info)

    def initialize(self, auto_discover: bool = True, metrics_dir: str = "src/metrics") -> None:
        """
        Initialize the registry by registering all available metrics.

        Args:
            auto_discover (bool): Whether to auto-discover metrics from files
            metrics_dir (str): Directory to search for metrics
        """
        if self._initialized:
            return

        print("🚀 Initializing metrics registry...")

        # Register built-in metrics
        self._register_builtin_metrics()

        # Auto-discover additional metrics if requested
        if auto_discover:
            self.auto_discover_metrics(metrics_dir)

        self._initialized = True
        print(f"✅ Registry initialized with {len(self._metrics)} metrics")

    def get_all_metrics(self) -> Dict[str, MetricInfo]:
        """
        Get all registered metrics.

        Returns:
            Dict[str, MetricInfo]: Dictionary mapping metric names to their information
        """
        if not self._initialized:
            self.initialize()
        return self._metrics.copy()

    def get_metric_info(self, name: str) -> Optional[MetricInfo]:
        """
        Get information about a specific metric.

        Args:
            name (str): Name of the metric

        Returns:
            Optional[MetricInfo]: Metric information if found, None otherwise
        """
        if not self._initialized:
            self.initialize()
        return self._metrics.get(name)

    def get_metric_names(self) -> List[str]:
        """
        Get list of all available metric names.

        Returns:
            List[str]: List of metric names
        """
        if not self._initialized:
            self.initialize()
        return list(self._metrics.keys())

    def get_script_paths(self) -> Dict[str, str]:
        """
        Get mapping of metric names to their script paths.

        Returns:
            Dict[str, str]: Dictionary mapping metric names to script paths
        """
        if not self._initialized:
            self.initialize()
        return {name: info.script_path for name, info in self._metrics.items()}

    def get_metrics_by_capability(
        self,
        reference_free: Optional[bool] = None,
        reference_based: Optional[bool] = None
    ) -> Dict[str, MetricInfo]:
        """
        Get metrics filtered by their capabilities.

        Args:
            reference_free (Optional[bool]): Filter by reference-free support
            reference_based (Optional[bool]): Filter by reference-based support

        Returns:
            Dict[str, MetricInfo]: Filtered metrics dictionary
        """
        if not self._initialized:
            self.initialize()

        filtered = {}
        for name, info in self._metrics.items():
            include = True

            if reference_free is not None:
                include = include and (info.supports_reference_free == reference_free)
            if reference_based is not None:
                include = include and (info.supports_reference_based == reference_based)

            if include:
                filtered[name] = info

        return filtered

    def debug_dependency_imports(self, metric_name: str) -> None:
        """
        Debug dependency imports by showing what actually works.

        Args:
            metric_name (str): Name of the metric to debug
        """
        metric_info = self.get_metric_info(metric_name)
        if not metric_info:
            print(f"❌ Metric {metric_name} not found in registry")
            return

        print(f"\n🔍 Debugging imports for {metric_name}:")
        print(f"Dependencies: {metric_info.dependencies}")

        for dep in metric_info.dependencies:
            print(f"\n📦 Testing package: {dep}")

            # Try various import patterns
            test_imports = [
                dep,
                dep.replace('-', '_'),
                dep.replace('-', '.'),
                dep.split('-')[-1],  # Last part after dash
                dep.split('-')[0],   # First part before dash
            ]

            # Add known mappings
            known_mappings = {
                'unbabel-comet': ['comet'],
                'python-dotenv': ['dotenv'],
                'scikit-learn': ['sklearn'],
                'opencv-python': ['cv2'],
                'pillow': ['PIL'],
                'beautifulsoup4': ['bs4'],
                'pyyaml': ['yaml']
            }

            if dep in known_mappings:
                test_imports.extend(known_mappings[dep])

            # Remove duplicates while preserving order
            seen = set()
            unique_imports = []
            for imp in test_imports:
                if imp not in seen:
                    seen.add(imp)
                    unique_imports.append(imp)

            found_working = []
            for import_name in unique_imports:
                try:
                    importlib.import_module(import_name)
                    found_working.append(import_name)
                    print(f"  ✅ import {import_name}")
                except ImportError as e:
                    print(f"  ❌ import {import_name} - {str(e)[:50]}...")

            if found_working:
                print(f"  🎯 Working imports: {found_working}")
            else:
                print(f"  ⚠️  No working imports found - package might not be installed")

    def check_dependencies(self, metric_name: str) -> Dict[str, bool]:
        """
        Check if dependencies for a metric are available.

        Args:
            metric_name (str): Name of the metric to check

        Returns:
            Dict[str, bool]: Dictionary mapping dependency names to availability status
        """
        metric_info = self.get_metric_info(metric_name)
        if not metric_info:
            return {}

        # Mapping from pip package names to import names
        package_import_map = {
            'unbabel-comet': 'comet',
            'python-dotenv': 'dotenv',
            'scikit-learn': 'sklearn',
            'opencv-python': 'cv2',
            'pillow': 'PIL',
            'beautifulsoup4': 'bs4',
            'pyyaml': 'yaml',
            'protobuf': 'google.protobuf',
            # Add more mappings as needed
        }

        dependency_status = {}
        for dep in metric_info.dependencies:
            # Get the actual import name
            import_name = package_import_map.get(dep, dep)

            # Try common variations if direct import fails
            import_variations = [
                import_name,
                import_name.replace('-', '_'),
                import_name.replace('-', '.'),
                dep.replace('-', '_'),
                dep.replace('-', '.')
            ]

            found = False
            for variation in import_variations:
                try:
                    importlib.import_module(variation)
                    dependency_status[dep] = True
                    found = True
                    break
                except ImportError:
                    continue

            if not found:
                dependency_status[dep] = False

        return dependency_status

    def get_available_metrics(self, check_dependencies: bool = False) -> List[str]:
        """
        Get list of metrics that are available (optionally checking dependencies).

        Args:
            check_dependencies (bool): Whether to check if dependencies are installed

        Returns:
            List[str]: List of available metric names
        """
        if not check_dependencies:
            return self.get_metric_names()

        available = []
        for name in self.get_metric_names():
            deps = self.check_dependencies(name)
            if all(deps.values()):  # All dependencies available
                available.append(name)

        return available

    def print_registry_info(self) -> None:
        """Print detailed information about all registered metrics."""
        if not self._initialized:
            self.initialize()

        print("\n" + "=" * 80)
        print("📊 METRICS REGISTRY")
        print("=" * 80)

        for name, info in self._metrics.items():
            print(f"\n🔧 {name}")
            print(f"   Class: {info.class_name}")
            print(f"   Module: {info.module_path}")
            print(f"   Script: {info.script_path}")
            print(f"   Dependencies: {', '.join(info.dependencies)}")
            print(f"   Reference-based: {info.supports_reference_based}")
            print(f"   Reference-free: {info.supports_reference_free}")
            if info.description:
                print(f"   Description: {info.description}")

        print("=" * 80)


# Global registry instance
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """
    Get the global metrics registry instance.

    Returns:
        MetricsRegistry: The global registry instance
    """
    return _registry


def get_available_metrics() -> Dict[str, str]:
    """
    Get mapping of available metrics to their script paths.
    This function maintains backward compatibility with existing code.

    Returns:
        Dict[str, str]: Dictionary mapping metric names to script paths
    """
    return get_registry().get_script_paths()


def initialize_registry(auto_discover: bool = True, metrics_dir: str = "src/metrics") -> None:
    """
    Initialize the global metrics registry.

    Args:
        auto_discover (bool): Whether to auto-discover metrics
        metrics_dir (str): Directory to search for metrics
    """
    get_registry().initialize(auto_discover, metrics_dir)


if __name__ == "__main__":
    # Demo/test the registry
    registry = get_registry()
    registry.initialize()
    registry.print_registry_info()
