import json
from torch.utils.data import Dataset

class QwenDataset(Dataset):
    def __init__(self, tokenizer, data, model_path):
        self.tokenizer = tokenizer
        self.model_path = model_path
        if isinstance(data, list) and all("conversations" in item for item in data):
            self.samples = self._process_conversation_data(data)
        else:
            self.samples = self._process_qa_data(data)

    def _process_qa_data(self, data):
        return [
            self.tokenizer(
                f"用户：{item['question']}\n助手：{item['answer']}",
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            for item in data
        ]

    def _process_conversation_data(self, data):
        samples = []
        for item in data:
            text = ""
            for turn in item["conversations"]:
                text += f"{turn['role']}：{turn['content']}\n"
            samples.append(
                self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = {k: v.squeeze() for k, v in self.samples[idx].items()}
        item["labels"] = item["input_ids"].clone()
        return item

def get_dataset(tokenizer, data_path, model_path):
    with open(data_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("数据应为JSON数组")
            return QwenDataset(tokenizer, data, model_path)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}")
        except KeyError as e:
            raise ValueError(f"数据字段缺失: {e}")