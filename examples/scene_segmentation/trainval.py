import torch

from vision3d_engine import IterBasedTrainer
from vision3d_engine.utils.parser import get_parser
from vision3d_engine.utils.optimizer import build_optimizer, build_scheduler
from vision3d_engine.utils.profiling import profile_cpu_runtime
from vision3d_engine.utils.tensor import tensor_to_array
from easy_kpconv.metrics import IntersectOverUnionMeter, AccuracyMeter

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import LossFunction, compute_metrics


def inject_parser():
    parser = get_parser()
    parser.add_argument("--test_area", default="Area_5", help="test area")


class Trainer(IterBasedTrainer):
    def __init__(self, cfg):
        inject_parser()

        super().__init__(cfg)

        # dataloader
        assert (
            cfg.model.neighbor_limits is not None
        ), "'neighbor_limits' is not set. Run 'calibrate_neighbors.py' and fill 'neighbor_limits' in 'config.py'."
        with profile_cpu_runtime("Data loader created: "):
            train_loader, val_loader = train_valid_data_loader(cfg, self.args.test_area)
        self.register_loader(train_loader, val_loader)

        # model
        model = create_model(cfg).cuda()
        model = self.register_model(model)

        # optimizer, scheduler
        optimizer = build_optimizer(model, cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = LossFunction(cfg)

        # select best model
        self.save_best_model_on_metric("SceneAcc")
        self.save_best_model_on_metric("mIoU")
        self.save_best_model_on_metric("mAcc")

        # meter
        self.num_classes = cfg.data.num_classes
        self.class_names = cfg.data.class_names
        self.iou_meter = None
        self.accuracy_meter = None

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = compute_metrics(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def before_val_epoch(self, epoch):
        self.iou_meter = IntersectOverUnionMeter(self.num_classes)
        self.accuracy_meter = AccuracyMeter(self.num_classes)

    def val_step(self, epoch, iteration, data_dict):
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

        # scene accuracy
        labels = data_dict["raw_labels"]
        accuracy = torch.eq(outputs, labels).float().mean()

        # mean iou & accuracy
        outputs = tensor_to_array(outputs)
        labels = tensor_to_array(labels)
        self.iou_meter.update(outputs, labels)
        self.accuracy_meter.update(outputs, labels)

        output_dict = {"outputs": outputs}
        result_dict = {"SceneAcc": accuracy}

        return output_dict, result_dict

    def after_val_epoch(self, epoch, summary_dict):
        mean_iou = self.iou_meter.mean_iou()
        mean_accuracy = self.accuracy_meter.mean_accuracy()
        self.log(f"MeanIOU: {mean_iou:.3f}", level="SUCCESS")
        self.log(f"MeanAcc: {mean_accuracy:.3f}", level="SUCCESS")
        summary_dict["mIoU"] = mean_iou
        summary_dict["mAcc"] = mean_accuracy


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
