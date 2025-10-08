import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
    Transform, Tuner, Trainer, Evaluator, Pusher
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def create_components(data_root, transform_module_file, trainer_module_file,
                      tuner_module_file, serving_model_dir):
    """
    Membuat semua komponen dalam pipeline TFX untuk proyek Drug Classification.

    Args:
        data_root (str): Path ke direktori yang berisi dataset CSV.
        transform_module_file (str): Path ke file Python fungsi preprocessing.
        trainer_module_file (str): Path ke file Python fungsi pelatihan model.
        tuner_module_file (str): Path ke file Python tuning hyperparameter.
        serving_model_dir (str): Direktori untuk menyimpan model serving.

    Returns:
        list: Daftar komponen TFX yang akan digunakan dalam pipeline.
    """
    # 1. EXAMPLEGEN: Membaca data dari file CSV
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    example_gen = CsvExampleGen(
        input_base=data_root,
        output_config=output_config
    )

    # 2. STATISTICSGEN: Menghasilkan statistik deskriptif
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # 3. SCHEMAGEN: Menghasilkan skema otomatis
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics']
    )

    # 4. EXAMPLEVALIDATOR: Memvalidasi data terhadap skema
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # 5. TRANSFORM: Melakukan transformasi fitur dan preprocessing
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file
    )

    # 6. TUNER: Melakukan hyperparameter tuning
    tuner = Tuner(
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        module_file=tuner_module_file,
        train_args=trainer_pb2.TrainArgs(num_steps=500),
        eval_args=trainer_pb2.EvalArgs(num_steps=100)
    )

    # 7. TRAINER: Melatih model menggunakan hasil tuning terbaik
    trainer = Trainer(
        module_file=trainer_module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=500)
    )

    # 8. RESOLVER: Mengambil model terbaik sebelumnya untuk perbandingan
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    # 9. EVALUATOR: Mengevaluasi model baru terhadap baseline model
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(
            label_key='Drug',
            prediction_key='probabilities'
        )],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name='SparseCategoricalAccuracy'),
                    tfma.MetricConfig(
                        class_name='SparseCategoricalCrossentropy'),
                    *[tfma.MetricConfig(
                        class_name=metric,
                        config=f'{{"class_id": {i}}}'
                    )
                        for i in range(5)
                        for metric in ['Precision', 'Recall']]
                ],
                thresholds={
                    'sparse_categorical_accuracy': tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        )
                    )
                }
            )
        ]
    )

    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    # 10. PUSHER: Menyimpan model ke direktori serving jika lolos evaluasi
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    # Mengembalikan daftar semua komponen pipeline
    return [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    ]
