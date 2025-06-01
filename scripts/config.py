from detectron2.config import get_cfg
from detectron2 import model_zoo


def setup(output_folder='', device='cuda'):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MIN_SIZE_TRAIN = (416,)
    cfg.INPUT.MAX_SIZE_TRAIN = 416
    cfg.INPUT.MIN_SIZE_TEST = 416
    cfg.INPUT.MAX_SIZE_TEST = 416
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ("dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20_000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # MUST BE EQUAL TO CLASSES
    cfg.OUTPUT_DIR = output_folder
    cfg.TEST.EVAL_PERIOD = 1_500
    cfg.SOLVER.CHECKPOINT_PERIOD = 1_500

    cfg.MODEL.DEVICE = device

    return cfg

