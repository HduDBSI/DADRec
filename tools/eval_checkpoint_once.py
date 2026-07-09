#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DADRec")
    parser.add_argument("--dataset", type=str, default="AmazonReviews2014")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--result_tsv", default=None)
    parser.add_argument("--run_name", required=True)
    args, unparsed = parser.parse_known_args()
    return args, parse_command_line_args(unparsed)


def to_builtin(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    return value


def main():
    args, command_line_configs = parse_args()
    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs,
    )

    test_dataloader = DataLoader(
        pipeline.tokenized_datasets["test"],
        batch_size=pipeline.config["eval_batch_size"],
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn["test"],
    )

    pipeline.accelerator.wait_for_everyone()
    pipeline.model, test_dataloader = pipeline.accelerator.prepare(
        pipeline.model, test_dataloader
    )
    pipeline.trainer.model.generate_w_decoding_graph = True
    test_results = pipeline.trainer.evaluate(test_dataloader)
    test_results = to_builtin(test_results)
    pipeline.log(f"Test Results: {test_results}")

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "metrics": test_results,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.result_tsv:
        tsv_path = Path(args.result_tsv)
        line = "\t".join(
            [
                args.run_name,
                str(test_results.get("ndcg@5", "")),
                str(test_results.get("ndcg@10", "")),
                str(test_results.get("recall@5", "")),
                str(test_results.get("recall@10", "")),
                str(test_results.get("weighted_score", "")),
                args.checkpoint,
                str(result_path),
            ]
        )
        with tsv_path.open("a") as f:
            f.write(line + "\n")

    pipeline.trainer.end()


if __name__ == "__main__":
    main()
