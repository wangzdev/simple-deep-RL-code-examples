import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

EPSILON = 1e-8


class DPODataset(Dataset):
    """Loads preference pairs from a JSONL file.

    Each line must be a JSON object with three string fields:
      - 'prompt'   : the shared input prompt
      - 'chosen'   : the preferred response
      - 'rejected' : the dispreferred response
    """

    def __init__(self, dataset_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        chosen_enc = self.tokenizer(
            prompt + chosen,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        rejected_enc = self.tokenizer(
            prompt + rejected,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        prompt_len = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].shape[1]

        return {
            'chosen_input_ids': chosen_enc['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_enc['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_enc['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_enc['attention_mask'].squeeze(0),
            'prompt_len': prompt_len
        }


class DPO:
    def __init__(
            self,
            model_name: str,
            beta: float = 0.1,
            lr: float = 1e-5,
            device: str = 'cpu'
    ):
        self.beta = beta
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # policy model (trained)
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        self.policy.to(device)

        # reference model (frozen)
        self.reference = AutoModelForCausalLM.from_pretrained(model_name)
        self.reference.to(device)
        for param in self.reference.parameters():
            param.requires_grad = False

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)

    def _compute_sequence_log_probs(self, model, input_ids, attention_mask, prompt_len):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)

        # shift so token i predicts token i+1
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # only score the response tokens, not the prompt or padding
        response_mask = shift_mask.clone().float()
        for i, p_len in enumerate(prompt_len):
            response_mask[i, :p_len - 1] = 0.0  # -1 because of the shift

        seq_log_probs = (token_log_probs * response_mask).sum(dim=-1) / (
            response_mask.sum(dim=-1) + EPSILON
        )
        return seq_log_probs

    def compute_dpo_loss(self, batch):
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        prompt_len = batch['prompt_len']

        policy_chosen_logps = self._compute_sequence_log_probs(
            self.policy, chosen_input_ids, chosen_attention_mask, prompt_len
        )
        policy_rejected_logps = self._compute_sequence_log_probs(
            self.policy, rejected_input_ids, rejected_attention_mask, prompt_len
        )

        with torch.no_grad():
            ref_chosen_logps = self._compute_sequence_log_probs(
                self.reference, chosen_input_ids, chosen_attention_mask, prompt_len
            )
            ref_rejected_logps = self._compute_sequence_log_probs(
                self.reference, rejected_input_ids, rejected_attention_mask, prompt_len
            )

        # DPO objective: maximise the margin between chosen and rejected relative to the reference
        chosen_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_ratio = policy_rejected_logps - ref_rejected_logps
        loss = -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio)).mean()
        return loss

    def train_step(self, batch):
        self.policy.train()
        loss = self.compute_dpo_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def train(
        dataset_path: str,
        model_name: str = 'gpt2',
        num_epochs: int = 3,
        batch_size: int = 4,
        lr: float = 1e-5,
        beta: float = 0.1,
        max_length: int = 512,
        device: str = 'cpu'
):
    trainer = DPO(model_name=model_name, beta=beta, lr=lr, device=device)

    dataset = DPODataset(dataset_path, trainer.tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            loss = trainer.train_step(batch)
            total_loss += loss
            if (batch_idx + 1) % 10 == 0:
                print(f'epoch {epoch}, batch {batch_idx}, loss: {loss:.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'epoch {epoch} complete, avg loss: {avg_loss:.4f}')
