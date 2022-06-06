import torch

from vision3d_engine import SingleTester
from vision3d_engine.utils.tensor import tensor_to_array
from vision3d_engine.utils.parser import get_parser
from vision3d_engine.utils.profiling import profile_cpu_runtime
from easy_kpconv.metrics import IntersectOverUnionMeter, AccuracyMeter

from dataset import test_data_loader
from config import make_cfg
from model import create_model
from loss import LossFunction


def inject_parser():
    parser = get_parser()
    parser.add_argument("--test_area", default="Area_5", help="test area")


class Tester(SingleTester):
    def __init__(self, cfg):
        inject_parser()

        super().__init__(cfg)

        # dataloader
        assert (
            cfg.model.neighbor_limits is not None
        ), "'neighbor_limits' is not set. Run 'calibrate_neighbors.py' and fill 'neighbor_limits' in 'config.py'."
        with profile_cpu_runtime("Data loader created"):
            data_loader = test_data_loader(cfg, self.args.test_area)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.loss_func = LossFunction(cfg)

        # misc
        self.num_classes = cfg.data.num_classes
        self.class_names = cfg.data.class_names
        self.iou_meter = IntersectOverUnionMeter(self.num_classes)
        self.accuracy_meter = AccuracyMeter(self.num_classes)

    def test_step(self, iteration, data_dict):
        # batch-wise prediction & average voting
        batch_list = data_dict["batch_list"]
        num_points = data_dict["labels"].shape[0]
        overall_scores = torch.zeros(size=(num_points, self.num_classes)).cuda()  # (N, C)
        for batch_dict in batch_list:
            output_dict = self.model(batch_dict)
            scores = output_dict["scores"]
            scores = torch.softmax(scores, dim=1)  # (M, C)
            indices = batch_dict["indices"].unsqueeze(1).expand_as(scores)  # (M, C)
            overall_scores.scatter_add_(dim=0, index=indices, src=scores)
        outputs = overall_scores.argmax(dim=1)  # (N,)
        # voxel to point
        inv_indices = data_dict["inv_indices"]
        outputs = outputs[inv_indices]
        return {"outputs": outputs}

    def eval_step(self, iteration, data_dict, output_dict):
        # scene accuracy
        outputs = output_dict["outputs"]
        labels = data_dict["raw_labels"]
        accuracy = torch.eq(outputs, labels).float().mean()

        # mean iou & accuracy
        outputs = tensor_to_array(outputs)
        labels = tensor_to_array(labels)
        self.iou_meter.update(outputs, labels)
        self.accuracy_meter.update(outputs, labels)

        return {"SceneAcc": accuracy}

    def after_test_epoch(self, summary_dict):
        message = f"MeanIOU: {self.iou_meter.mean_iou():.3f}"
        self.log(message, level="SUCCESS")
        for i in range(self.num_classes):
            message = f"  {self.class_names[i]}: {self.iou_meter.iou(i):.3f}"
            self.log(message, level="SUCCESS")

        message = f"MeanAcc: {self.accuracy_meter.mean_accuracy():.3f}"
        self.log(message, level="SUCCESS")
        for i in range(self.num_classes):
            message = f"  {self.class_names[i]}: {self.accuracy_meter.accuracy(i):.3f}"
            self.log(message, level="SUCCESS")


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
