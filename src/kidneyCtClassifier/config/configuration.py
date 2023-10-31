import os
from kidneyCtClassifier.constants import *
from kidneyCtClassifier.entity.config_entity import DataIngestionConfig, TrainingConfig, PrepareBaseModelConfig
from kidneyCtClassifier.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        training_data = os.path.join(self.config.data_ingestion.root_dir, 'kidney-ct-scan-image')
        create_directories([config.root_dir])

        data_ingestion_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_learning_rate=self.params.LR,
            params_epochs=self.params.N_EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
        )

        return data_ingestion_config
