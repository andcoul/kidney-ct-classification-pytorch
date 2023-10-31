from kidneyCtClassifier.config.configuration import ConfigurationManager
from kidneyCtClassifier.components.model_preparation import ModelPreparation
from kidneyCtClassifier import logger

STAGE_NAME = "Model Preparation"


class ModelPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = ModelPreparation(config=base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.full_model_updated()


if __name__ == "__main__":
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
