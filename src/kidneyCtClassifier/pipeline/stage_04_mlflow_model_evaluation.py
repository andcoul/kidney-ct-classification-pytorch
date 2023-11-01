from kidneyCtClassifier.config.configuration import ConfigurationManager
from kidneyCtClassifier.components.mlflow_model_evaluation import MlflowModelEvaluation
from kidneyCtClassifier import logger

STAGE_NAME = "Mlflow Evaluation"


class MlflowModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_eval_config = config.get_evaluation_config()
        mlflow_evaluation = MlflowModelEvaluation(config=model_eval_config)
        mlflow_evaluation.inference()
        # mlflow_evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = MlflowModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
