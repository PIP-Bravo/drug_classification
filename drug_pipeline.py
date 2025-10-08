from modules.components import create_components
import os
import sys
from absl import logging

import tensorflow as tf
from tfx.orchestration import pipeline
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

# Menambahkan direktori 'modules' ke dalam sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))


# Mengatur tingkat verbosity logging
logging.set_verbosity(logging.INFO)


def create_pipeline():
    """
    Membuat dan mengonfigurasi TFX pipeline untuk proyek Drug Classification.

    Pipeline ini terdiri dari beberapa komponen TFX seperti ExampleGen,
    SchemaGen, Transform, Tuner, Trainer, Evaluator, dan Pusher.

    Returns:
        beam_pipeline (tfx.orchestration.pipeline.Pipeline):
            Objek pipeline yang siap dijalankan oleh BeamDagRunner.
    """
    # Konfigurasi dasar lokasi file dan direktori proyek
    BASE_DIR = r"C:\drug-pipeline"
    DATA_ROOT = os.path.join(BASE_DIR, "data")  # Lokasi dataset mentah
    MODULE_DIR = os.path.join(BASE_DIR, "modules")  # Lokasi modul kustom
    PIPELINE_ROOT = os.path.join(BASE_DIR, "output", "pipeline_output")
    SERVING_MODEL_DIR = os.path.join(BASE_DIR, "output", "serving_model")
    METADATA_PATH = os.path.join(BASE_DIR, "output", "metadata.sqlite")

    # Lokasi file module TFX
    TRANSFORM_MODULE_FILE = os.path.join(MODULE_DIR, "drug_class_transform.py")
    TRAINER_MODULE_FILE = os.path.join(MODULE_DIR, "drug_class_trainer.py")
    TUNER_MODULE_FILE = os.path.join(MODULE_DIR, "drug_class_tuner.py")

    # Membuat direktori yang diperlukan bila belum ada
    os.makedirs(PIPELINE_ROOT, exist_ok=True)
    os.makedirs(SERVING_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    # Membuat daftar komponen TFX dengan fungsi helper kustom
    components = create_components(
        data_root=DATA_ROOT,
        transform_module_file=TRANSFORM_MODULE_FILE,
        trainer_module_file=TRAINER_MODULE_FILE,
        tuner_module_file=TUNER_MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR
    )

    # Konfigurasi koneksi metadata
    metadata_config = sqlite_metadata_connection_config(METADATA_PATH)

    # Menyusun pipeline TFX menggunakan Beam sebagai orchestrator
    beam_pipeline = pipeline.Pipeline(
        pipeline_name="drug_classification_pipeline_with_tuner",
        pipeline_root=PIPELINE_ROOT,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_config,
        beam_pipeline_args=['--direct_running_mode=multi_processing']
    )

    return beam_pipeline


if __name__ == "__main__":
    """Entry point utama untuk menjalankan pipeline Drug Classification."""
    print("Starting Drug Classification Pipeline")

    # Menyembunyikan pesan warning atau log berlebihan dari TensorFlow
    tf.get_logger().setLevel('ERROR')

    try:
        # Membuat objek pipeline dan menjalankannya dengan Beam
        pipeline_obj = create_pipeline()
        BeamDagRunner().run(pipeline_obj)
        logging.info("Pipeline completed successfully!")

    except Exception as e:
        # Menangani error jika pipeline gagal dijalankan
        logging.error(f"Pipeline failed with error: {str(e)}")
        raise
