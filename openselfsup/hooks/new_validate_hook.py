from mmcv.runner import Hook
import numpy
import torch
import mmcv
import numpy as np
from torch.utils.data import Dataset
from openselfsup.models.byol import BYOL
from openselfsup.utils import nondist_forward_collect, dist_forward_collect
from .registry import HOOKS


@HOOKS.register_module
class NewValidateHook(Hook):
    """Validation hook.

    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        dist_mode (bool): Use distributed evaluation or not. Default: True.
        initial (bool): Whether to evaluate before the training starts.
            Default: True.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataset,
                 dist_mode=True,
                 initial=True,
                 interval=1,
                 **eval_kwargs):
        from openselfsup import datasets
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=False,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()),
        )
        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def before_run(self, runner):
        if self.initial:
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self._run_validate(runner)

    def _run_validate(self, runner):
        runner.model.eval()
        results = []
        prog_bar = mmcv.ProgressBar(len(self.data_loader))
        for idx, data in enumerate(self.data_loader):
            with torch.no_grad():
                result = runner.model(data['img'],mode='test')  # dict{key: tensor}
            results.append(result['loss'].cpu().numpy())
            prog_bar.update()

        eval_res = np.array(results).mean()
        print(eval_res)
        if runner.rank == 0:
            self._evaluate(runner, eval_res, 'loss')

        # runner.log_buffer.output['loss'] = eval_res
        # runner.log_buffer.ready = True
        runner.model.train()

    def _evaluate(self, runner, results, keyword):
        eval_res = self.dataset.evaluate(
            results,
            keyword=keyword,
            logger=runner.logger)
        for name,val in eval_res.items():
            runner.log_buffer.output[name] = str(val)
        runner.log_buffer.ready = True
