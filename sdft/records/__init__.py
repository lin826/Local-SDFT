"""Shared data collection and performance-benchmark contract for Local-SDFT.

CLI and web surfaces should import from this package rather than inventing
parallel schemas or storage locations.
"""

from .benchmark import (
    geek_jokes_generations_path,
    measure_chat,
    measure_generation,
    measure_geek_jokes,
    measure_inference,
    persist_performance_result,
    run_benchmark,
)
from .collect import (
    collect_record,
    export_collected_for_training,
    import_training_row,
    new_collected_record,
    validate_collected_record,
)
from .paths import (
    collected_export_path,
    collected_records_path,
    performance_dir,
    performance_index_path,
    performance_result_path,
    project_root,
)
from .schema import (
    SCHEMA_VERSION,
    CollectedRecord,
    PerformanceMetrics,
    PerformanceResult,
)
from .store import (
    append_collected_record,
    export_training_jsonl,
    list_performance_results,
    load_collected_records,
    load_performance_index,
    load_performance_result,
    save_performance_result,
)

__all__ = [
    "SCHEMA_VERSION",
    "CollectedRecord",
    "PerformanceMetrics",
    "PerformanceResult",
    "append_collected_record",
    "collect_record",
    "collected_export_path",
    "collected_records_path",
    "export_collected_for_training",
    "export_training_jsonl",
    "import_training_row",
    "list_performance_results",
    "load_collected_records",
    "load_performance_index",
    "load_performance_result",
    "geek_jokes_generations_path",
    "measure_chat",
    "measure_generation",
    "measure_geek_jokes",
    "measure_inference",
    "new_collected_record",
    "performance_dir",
    "performance_index_path",
    "performance_result_path",
    "persist_performance_result",
    "project_root",
    "run_benchmark",
    "save_performance_result",
    "validate_collected_record",
]
