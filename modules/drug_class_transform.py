import tensorflow as tf
import tensorflow_transform as tft

# Daftar fitur kategorikal beserta jumlah kategori unik
CATEGORICAL_FEATURES = {
    'Sex': 2,           # Male, Female
    'BP': 3,            # Low, Normal, High
    'Cholesterol': 2    # Normal, High
}

# Daftar fitur numerik
NUMERICAL_FEATURES = ['Age', 'Na_to_K']

# Nama kolom label (target)
LABEL_KEY = 'Drug'


def preprocessing_fn(inputs):
    """
    Fungsi preprocessing yang digunakan oleh komponen Transform TFX.

    Args:
        inputs (dict): Dictionary berisi input fitur mentah dari dataset TFX.

    Returns:
        dict: Dictionary berisi fitur-fitur yang telah ditransformasikan.
    """
    outputs = {}

    # Proses fitur numerik
    for feature in NUMERICAL_FEATURES:
        # Pastikan tipe data numerik (konversi jika masih string)
        if inputs[feature].dtype == tf.string:
            float_feature = tf.strings.to_number(
                inputs[feature], out_type=tf.float32)
        else:
            float_feature = tf.cast(inputs[feature], tf.float32)

        # Normalisasi fitur numerik dengan z-score
        outputs[feature + '_xf'] = tft.scale_to_z_score(float_feature)

    # Proses fitur kategorikal
    for feature, vocab_size in CATEGORICAL_FEATURES.items():
        # Pastikan fitur dikonversi ke string sebelum membuat vocabulary
        if inputs[feature].dtype != tf.string:
            string_feature = tf.as_string(inputs[feature])
        else:
            string_feature = inputs[feature]

        # Mapping nilai kategori ke indeks integer
        outputs[feature + '_xf'] = tft.compute_and_apply_vocabulary(
            string_feature,
            top_k=vocab_size,
            num_oov_buckets=1
        )

    # Proses label (Drug)
    if inputs[LABEL_KEY].dtype != tf.string:
        string_label = tf.as_string(inputs[LABEL_KEY])
    else:
        string_label = inputs[LABEL_KEY]

    # Transformasi label menjadi indeks integer
    outputs[LABEL_KEY] = tft.compute_and_apply_vocabulary(string_label)

    return outputs


def _transformed_name(key):
    """
    Mengembalikan nama fitur yang telah ditransformasi.

    Args:
        key (str): Nama fitur asli.

    Returns:
        str: Nama fitur hasil transformasi.
    """
    return key + '_xf'
