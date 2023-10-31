from kidneyCtClassifier import logger
from kidneyCtClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from kidneyCtClassifier.pipeline.stage_02_model_preparation import ModelPreparationPipeline
from kidneyCtClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"*******************")
    logger.info(f">>>> stage {STAGE_NAME} has started")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Preparation"

try:
    logger.info(f"*******************")
    logger.info(f">>>> stage {STAGE_NAME} has started")
    model_preparation = ModelPreparationPipeline()
    model_preparation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training"

try:
    logger.info(f"*******************")
    logger.info(f">>>> stage {STAGE_NAME} has started")
    model_training = ModelTrainingPipeline()
    model_training.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
