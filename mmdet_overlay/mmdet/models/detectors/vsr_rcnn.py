from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.faster_rcnn import FasterRCNN


@DETECTORS.register_module()
class VSRRCNN(FasterRCNN):
    """Faster R-CNN with the proposal location map forwarded to the RoI head."""

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        assert self.with_bbox, 'A bounding-box head is required.'
        features = self.extract_feat(img)

        if proposals is None:
            proposals, loc_preds = self.rpn_head.simple_center_test_rpn(
                features, img_metas)
        else:
            loc_preds = None

        return self.roi_head.simple_test(
            features,
            proposals,
            loc_preds,
            img_metas,
            rescale=rescale)
