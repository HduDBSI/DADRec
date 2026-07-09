import json
import os

from datasets import Dataset

from genrec.dataset import AbstractDataset


class LocalSeq(AbstractDataset):
    """Load MACRec-style `{dataset}.inter.json` sequential recommendation data."""

    def __init__(self, config: dict):
        super(LocalSeq, self).__init__(config)
        self.local_data_dir = config["local_data_dir"]
        self.local_dataset_name = config["local_dataset_name"]
        self._download_and_process_raw()

    def _download_and_process_raw(self):
        inter_path = os.path.join(self.local_data_dir, f"{self.local_dataset_name}.inter.json")
        if not os.path.exists(inter_path):
            raise FileNotFoundError(f"LocalSeq interactions not found: {inter_path}")
        with open(inter_path, "r") as f:
            inters = json.load(f)

        for raw_user, raw_items in inters.items():
            user = str(raw_user)
            if user not in self.id_mapping["user2id"]:
                self.id_mapping["user2id"][user] = len(self.id_mapping["id2user"])
                self.id_mapping["id2user"].append(user)
            seq = []
            for raw_item in raw_items:
                item = str(raw_item)
                if item not in self.id_mapping["item2id"]:
                    self.id_mapping["item2id"][item] = len(self.id_mapping["id2item"])
                    self.id_mapping["id2item"].append(item)
                seq.append(item)
            self.all_item_seqs[user] = seq

        item_path = os.path.join(self.local_data_dir, f"{self.local_dataset_name}.item.json")
        if os.path.exists(item_path):
            with open(item_path, "r") as f:
                self.item2meta = json.load(f)
        else:
            self.item2meta = {}

    def split(self):
        if self.split_data is not None:
            return self.split_data
        datasets = self._leave_one_out()
        sample_users = int(self.config.get("local_sample_users", 0) or 0)
        if sample_users > 0:
            limited = {}
            for split, data in datasets.items():
                keep = min(sample_users, len(data))
                limited[split] = Dataset.from_dict({
                    "user": data["user"][:keep],
                    "item_seq": data["item_seq"][:keep],
                })
            datasets = limited
        self.split_data = datasets
        return datasets
