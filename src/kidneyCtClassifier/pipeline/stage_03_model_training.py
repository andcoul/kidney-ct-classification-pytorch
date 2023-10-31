from kidneyCtClassifier.config.configuration import ConfigurationManager
from kidneyCtClassifier.components.model_training import DataTraining
from kidneyCtClassifier import logger

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_training_config = config.get_training_config()
        data_training = DataTraining(config=data_training_config)
        data_training.main()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
