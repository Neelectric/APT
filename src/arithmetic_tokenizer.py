# huge props to skyzip and jbm on https://stackoverflow.com/questions/69531811/using-hugginface-transformers-and-tokenizers-with-a-fixed-vocabulary
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

from transformers import PreTrainedTokenizer

class ArithmeticTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Union[Dict[str, int], str], max_length: int = None, padding=False):
        if isinstance(vocab, str):
            vocab_path = Path(vocab)
            with open(vocab_path, 'r') as f:
                self._token_ids = json.load(f)
        else:
            self._token_ids = vocab
        id_to_token = {value: key for key, value in self._token_ids.items()}
        self._id_tokens = id_to_token
        super().__init__(max_len=max_length)
        self.max_length = max_length
        if "<unk>" in id_to_token.values():
            self.unk_token = '<unk>'
            self.unk_token_id = self._token_ids.get(self.unk_token, 0)
        if "<pad>" in id_to_token.values():
            self.pad_token = '<pad>'
            self.pad_token_id = self._token_ids.get(self.pad_token, 1)
        if "<bos>" in id_to_token.values():
            self.bos_token = '<bos>'
            self.bos_token_id = self._token_ids.get(self.bos_token, 2)
        if "<eos>" in id_to_token.values():
            self.eos_token = '<eos>'
            self.eos_token_id = self._token_ids.get(self.eos_token, 3)
        if "<mask>" in id_to_token.values():
            self.mask_token = '<mask>'
            self.mask_token_id = self._token_ids.get(self.mask_token, 4)
        
        # self.padding=padding
        self.padding_side="left"

    def _tokenize(self, text: str, **kwargs):
        # return text.split('')
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_ids[token] if token in self._token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_tokens[index] if index in self._id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self._token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory, filename_prefix + 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(self._token_ids, f)
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self._token_ids)

# char_to_id = {}
# char_to_id["<bos>"] = 0

# for i in range(1,101):
#     char_to_id[str(i)] = i
# char_to_id["+"] = i + 1
# char_to_id["="] = i + 2
# char_to_id["<pad>"] = i + 3
# char_to_id["<unk>"] = i + 4
# char_to_id["<mask>"] = i + 5
# char_to_id["<eos>"] = i + 6

# with open('tokenizer/vocab.json', 'w') as f:
#     json.dump(char_to_id, f)

# print(char_to_id)






# sum_string_ex = "<bos>18+19=37<eos>"
# model_max_len = 10

# # Optionally specify the path to a vocab file
# vocab_path = 'tokenizer/sum_0-9+special_vocab.json'

# # You can either pass the custom vocab dictionary or the path to the vocab file
# tokenizer = APTTokenizer(vocab_path, max_len=model_max_len)

# res = tokenizer(
#     [
#         "<bos> 1 8 + 1 9 = 3 7 <eos>",
#         "<bos> 2 + 4 3 = 4 5 <eos>",
#     ],
#     # padding=True,
#     truncation=True,
# )
# print(res)