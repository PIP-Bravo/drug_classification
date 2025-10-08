from typing import List, Text

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt

from tfx.components.tuner.component import TunerFnResult
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options

# KONSTANTA FITUR
NUMERICAL_FEATURES = ['Age', 'Na_to_K']
CATEGORICAL_FEATURES = ['Sex', 'BP', 'Cholesterol']
LABEL_KEY = 'Drug'


def _transformed_name(key: str) -> str:
    """
    Menambahkan suffix '_xf' pada nama fitur untuk menunjukkan transformasi.

    Args:
        key (str): Nama kolom asli.

    Returns:
        str: Nama kolom setelah transformasi.
    """
    return key + '_xf'


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    """
    Membuat dataset TensorFlow dari file TFRecord hasil transformasi.

    Args:
        file_pattern (List[Text]): Daftar path TFRecord hasil transformasi.
        data_accessor (DataAccessor): Objek utilitas untuk membaca data.
        tf_transform_output (tft.TFTransformOutput): Output hasil transformasi.
        batch_size (int): Ukuran batch dataset.

    Returns:
        tf.data.Dataset: Dataset siap untuk training/evaluasi.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=LABEL_KEY
        ),
        tf_transform_output.transformed_metadata.schema
    )


def _build_keras_model(hp):
    """
    Membangun arsitektur model Keras untuk proses hyperparameter tuning.

    Args:
        hp (keras_tuner.HyperParameters): Objek hyperparameter.

    Returns:
        tf.keras.Model: Model Keras yang sudah dikompilasi dan siap dituning.
    """
    # Daftar hyperparameter yang akan di-tuning
    learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    units_layer_1 = hp.Int('units_layer_1', 64, 128, step=32)
    units_layer_2 = hp.Int('units_layer_2', 32, 64, step=16)
    dropout_rate = hp.Float('dropout_rate', 0.2, 0.4)
    embedding_dim = hp.Int('embedding_dim', 2, 4)

    # Input layer untuk setiap fitur
    inputs = {}
    feature_list = []

    # Fitur numerik langsung digunakan sebagai input dense
    for feature in NUMERICAL_FEATURES:
        transformed_name = _transformed_name(feature)
        inputs[transformed_name] = tf.keras.layers.Input(
            shape=(1,), name=transformed_name, dtype=tf.float32)
        feature_list.append(inputs[transformed_name])

    # Fitur kategorikal akan diubah menjadi embedding
    for feature in CATEGORICAL_FEATURES:
        transformed_name = _transformed_name(feature)
        inputs[transformed_name] = tf.keras.layers.Input(
            shape=(1,), name=transformed_name, dtype=tf.int64)

        # Embedding layer untuk representasi fitur kategorikal
        embedded = tf.keras.layers.Embedding(
            input_dim=10,
            output_dim=embedding_dim,
            name=f'{feature}_embedding'
        )(inputs[transformed_name])
        flattened = tf.keras.layers.Flatten()(embedded)
        feature_list.append(flattened)

    # Menggabungkan semua fitur
    combined = tf.keras.layers.concatenate(feature_list)

    # Hidden layers
    x = combined
    x = tf.keras.layers.Dense(units_layer_1, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units_layer_2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Kompilasi model menggunakan learning rate hasil tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    return model


def tuner_fn(fn_args: TrainerFnArgs) -> TunerFnResult:
    """
    Fungsi utama untuk mendefinisikan dan menjalankan hyperparameter tuning.

    Args:
        fn_args (TrainerFnArgs): Argumen dari komponen TFX Trainer/Tuner.

    Returns:
        TunerFnResult: Objek berisi konfigurasi tuner dan parameter fit.
    """
    # 1. Memuat hasil transformasi fitur
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # 2. Membuat dataset training dan evaluasi
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=32
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=32
    )

    # 3. Membuat objek tuner dengan strategi Random Search
    tuner = kt.RandomSearch(
        hypermodel=_build_keras_model,
        objective='val_sparse_categorical_accuracy',
        max_trials=3,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='drug_classification_tuning',
        overwrite=True
    )

    # 4. Mengambil jumlah langkah training dan evaluasi
    train_steps = fn_args.train_steps
    eval_steps = fn_args.eval_steps

    print(f"Training steps: {train_steps}")
    print(f"Evaluation steps: {eval_steps}")

    # 5. Mengembalikan konfigurasi tuner dan parameter training
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': train_steps,
            'validation_steps': eval_steps,
            'epochs': 5,
            'verbose': 1
        }
    )
