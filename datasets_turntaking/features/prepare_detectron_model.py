# Some basic setup:
import detectron2
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.layers.mask_ops import _do_paste_mask
from detectron2.utils.logger import setup_logger


### create model and predictor
def create_model(threshold = 0.3):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    model_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    
    # build predictor (for corner images)
    predictor = DefaultPredictor(cfg)

    # build model (for batch inference in closeup images)
    model_detectron = build_model(cfg)

    # get pretrained weight from model zoo
    DetectionCheckpointer(model_detectron).load(model_zoo.get_checkpoint_url(model_path)) 
    model_detectron.eval() # set to eval

    return model_detectron, predictor
