import torch
import json
import torch.backends.cudnn as cudnn

from config import args
from trainer import Trainer, RuleTrainer
from logger_config import logger


def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))
    json.dump(args.__dict__, open(args.result_config_path, 'w', encoding='UTF-8'), ensure_ascii=False, indent=4)

    logger.info('pre-training rule embedding')
    rule_trainer = RuleTrainer(args, ngpus_per_node=ngpus_per_node)
    rule_trainer.train_loop()
    rule_trainer.save_rule_head_vector()

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    trainer.train_loop()


if __name__ == '__main__':
    main()
