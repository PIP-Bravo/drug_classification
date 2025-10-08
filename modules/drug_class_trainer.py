from typing import List, Text, Dict, Any

import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options

# KONSTANTA FITUR
NUMERICAL_FEATURES = ['Age', 'Na_to_K']
CATEGORICAL_FEATURES = ['Sex', 'BP', 'Cholesterol']
LABEL_KEY = 'Drug'


def _transformed_name(key: Text) -> Text:
    """
    Menambahkan suffix '_xf' pada nama fitur hasil transformasi.

    Args:
        key (Text): Nama fitur asli.

    Returns:
        Text: Nama fitur hasil transformasi.
    """
    return key + '_xf'


def _get_serve_tf_examples_fn(model: tf.keras.Model,
                              tf_transform_output: tft.TFTransformOutput):
    """
    Membuat fungsi serving TensorFlow.

    Args:
        model (tf.keras.Model): Model terlatih TensorFlow.
        tf_transform_output (tft.TFTransformOutput): Output komponen Transform.

    Returns:
        Callable: Fungsi serving TensorFlow.
    """
    # Layer untuk mengubah data mentah menjadi fitur transformasi
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Fungsi untuk parsing dan transformasi input sebelum inferensi."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)  # Label tidak digunakan saat serving

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    """
    Membuat input dataset TensorFlow untuk training atau evaluasi model.

    Args:
        file_pattern (List[Text]): Daftar file input.
        data_accessor (DataAccessor): Utilitas akses data dari TFX.
        tf_transform_output (tft.TFTransformOutput): Output transformasi fitur.
        batch_size (int): Ukuran batch untuk dataset.

    Returns:
        tf.data.Dataset: Dataset TensorFlow siap untuk training/evaluasi.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=LABEL_KEY
        ),
        tf_transform_output.transformed_metadata.schema
    )


def _extract_hyperparameters(tuner_data: Dict[Text, Any]) -> Dict[Text, Any]:
    """
    Mengekstrak nilai hyperparameter dari struktur data hasil Keras Tuner.

    Args:
        tuner_data (Dict[Text, Any]): Data hasil tuning.

    Returns:
        Dict[Text, Any]: Dictionary berisi hyperparameter.
    """
    hp_values = {}

    # Berbagai kemungkinan struktur hasil Tuner
    if 'values' in tuner_data:
        hp_values = tuner_data['values']
        print(f"Extracted hyperparameters from 'values': {hp_values}")
    elif 'hyperparameters' in tuner_data:
        hp_values = tuner_data['hyperparameters']
        print(f"Extracted from 'hyperparameters': {hp_values}")
    elif 'best_hyperparameters' in tuner_data:
        hp_values = tuner_data['best_hyperparameters']
        print(f"Extracted from 'best_hyperparameters': {hp_values}")
    else:
        hp_values = tuner_data
        print(f"Using direct hyperparameters: {hp_values}")

    return hp_values


def _build_keras_model(
    hp_values: Dict[Text, Any],
    tf_transform_output: tft.TFTransformOutput
) -> tf.keras.Model:

    """
    Membangun model Keras menggunakan hyperparameter terbaik dari Tuner.

    Args:
        hp_values (Dict[Text, Any]): Nilai-nilai hyperparameter hasil tuning.
        tf_transform_output (tft.TFTransformOutput): Output transformasi fitur.

    Returns:
        tf.keras.Model: Model Keras siap untuk training.
    """
    print(f"Building model with hyperparameters: {hp_values}")

    # Ambil hyperparameter utama
    learning_rate = hp_values['learning_rate']
    dropout_rate = hp_values['dropout_rate']
    embedding_dim = hp_values['embedding_dim']
    units_layer_1 = hp_values.get('units_layer_1', 64)
    units_layer_2 = hp_values.get('units_layer_2', 32)
    hidden_units = [units_layer_1, units_layer_2]

    print(
        f"Model architecture: hidden_units={hidden_units}, "
        f"dropout={dropout_rate}, "
        f"embed_dim={embedding_dim}, "
        f"lr={learning_rate}"
    )

    # Input layers
    inputs, feature_list = {}, []

    # Fitur numerik
    for feature in NUMERICAL_FEATURES:
        transformed_name = _transformed_name(feature)
        inputs[transformed_name] = tf.keras.layers.Input(
            shape=(1,), name=transformed_name, dtype=tf.float32)
        feature_list.append(inputs[transformed_name])

    # Fitur kategorikal (menggunakan embedding)
    for feature in CATEGORICAL_FEATURES:
        transformed_name = _transformed_name(feature)
        inputs[transformed_name] = tf.keras.layers.Input(
            shape=(1,), name=transformed_name, dtype=tf.int64)

        embedded = tf.keras.layers.Embedding(
            input_dim=10,  # jumlah kategori maksimum
            output_dim=embedding_dim,
            name=f'{feature}_embedding'
        )(inputs[transformed_name])

        feature_list.append(tf.keras.layers.Flatten()(embedded))

    # Gabungkan semua fitur
    combined = tf.keras.layers.concatenate(feature_list)

    # Hidden layers
    x = combined
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    # Bangun dan compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    model.summary()
    return model


def run_fn(fn_args: TrainerFnArgs):
    """
    Fungsi utama untuk proses training model menggunakan TFX Trainer Component.

    Args:
        fn_args (TrainerFnArgs): Argumen dari komponen Trainer TFX.
    """
    # Load hasil transformasi fitur
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Muat hyperparameter terbaik
    hp_values = {}
    if hasattr(fn_args, 'hyperparameters') and fn_args.hyperparameters:
        hp_values = _extract_hyperparameters(fn_args.hyperparameters)
        print(f"Using hyperparameters from fn_args: {hp_values}")

    # Validasi keberadaan hyperparameter
    if not hp_values:
        raise ValueError("No hyperparameters found.")

    required_params = ['learning_rate', 'dropout_rate', 'embedding_dim']
    missing_params = [param for param in required_params
                      if param not in hp_values]

    if missing_params:
        raise ValueError(f"Missing required hyperparameters: {missing_params}")

    print(f"Hyperparameters for training: {hp_values}")

    # Load dataset training dan evaluasi
    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor,
        tf_transform_output, batch_size=32
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor,
        tf_transform_output, batch_size=32
    )

    # Bangun model berdasarkan hyperparameter
    model = _build_keras_model(hp_values, tf_transform_output)

    # Callback early stopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]

    # Proses training
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    # Simpan model dengan fungsi serving
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)
    print("Model trained and saved successfully!")
