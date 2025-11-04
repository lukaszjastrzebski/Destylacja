"""
Główny skrypt treningowy do destylacji wiedzy z dużego modelu do małego.
Implementuje mieszane straty (KL divergence + Cross Entropy) oraz LoRA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset
import numpy as np
from tqdm import tqdm


@dataclass
class DistillationConfig:
    """Konfiguracja procesu destylacji."""
    kl_weight: float = 0.5
    ce_weight: float = 0.5
    temperature: float = 2.0


class DistillationTrainer(Trainer):
    """
    Custom Trainer implementujący destylację wiedzy.
    Łączy stratę KL divergence (od nauczyciela) z Cross Entropy (od prawdziwych etykiet).
    """
    
    def __init__(
        self,
        teacher_model,
        distillation_config: DistillationConfig,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.distillation_config = distillation_config
        
        # Zamroź parametry nauczyciela
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Oblicza mieszaną stratę: KL divergence + Cross Entropy.
        
        KL divergence zachowuje "miękkie" odpowiedzi nauczyciela (styl).
        Cross Entropy zapewnia poprawność odpowiedzi (sens).
        """
        labels = inputs.pop("labels")
        
        # Forward pass przez studenta
        student_outputs = model(**inputs, output_hidden_states=True)
        student_logits = student_outputs.logits
        
        # Forward pass przez nauczyciela
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits
        
        # 1. Cross Entropy Loss (standard language modeling loss)
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # 2. KL Divergence Loss (destylacja od nauczyciela)
        # Używamy temperatury aby "zmiękcz" rozkłady prawdopodobieństwa
        temperature = self.distillation_config.temperature
        
        # Soft targets od nauczyciela
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        # Soft predictions od studenta
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL divergence (tylko dla nie-padding tokenów)
        loss_kl = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='batchmean'
        ) * (temperature ** 2)  # Skalowanie przez T^2 (standardowa praktyka)
        
        # 3. Kombinowana strata
        loss = (
            self.distillation_config.ce_weight * loss_ce +
            self.distillation_config.kl_weight * loss_kl
        )
        
        # Logowanie do wandb/tensorboard
        self.log({
            'loss_ce': loss_ce.item(),
            'loss_kl': loss_kl.item(),
            'loss_total': loss.item(),
        })
        
        return (loss, student_outputs) if return_outputs else loss


class NPCDistillation:
    """Główna klasa zarządzająca procesem destylacji."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Inicjalizacja procesu destylacji."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.teacher_config = self.config['teacher_model']
        self.student_config = self.config['student_model']
        self.training_config = self.config['training']
        
        print("=" * 60)
        print("Inicjalizacja destylacji LLM do NPC")
        print("=" * 60)
        
        # Załaduj tokenizer (wspólny dla obu modeli)
        self.load_tokenizer()
        
        # Załaduj modele
        self.load_teacher_model()
        self.load_student_model()
        
    def load_tokenizer(self):
        """Ładuje ujednolicony tokenizer."""
        print(f"\n📝 Ładowanie tokenizera...")
        
        # Używamy tokenizera od studenta (lub nauczyciela - powinny być zgodne)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.student_config['name'],
            trust_remote_code=True
        )
        
        # Ustawienie pad_token jeśli nie istnieje
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer załadowany (vocab size: {len(self.tokenizer)})")
    
    def load_teacher_model(self):
        """Ładuje model nauczyciela."""
        print(f"\n🎓 Ładowanie modelu nauczyciela: {self.teacher_config['name']}")
        
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        
        if self.teacher_config.get('load_in_8bit', False):
            model_kwargs['load_in_8bit'] = True
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.teacher_config['name'],
            **model_kwargs
        )
        self.teacher_model.eval()
        
        print(f"✓ Model nauczyciela załadowany")
        print(f"  Parametry: {sum(p.numel() for p in self.teacher_model.parameters()) / 1e9:.2f}B")
    
    def load_student_model(self):
        """Ładuje model studenta i konfiguruje LoRA."""
        print(f"\n🎒 Ładowanie modelu studenta: {self.student_config['name']}")
        
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.student_config['name'],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,  # Oszczędność pamięci
        )
        
        # Przygotowanie modelu do treningu z kwantyzacją (tylko jeśli teacher używa 8bit)
        if self.teacher_config.get('load_in_8bit', False):
            self.student_model = prepare_model_for_kbit_training(self.student_model)
        else:
            # Gradient checkpointing bez kwantyzacji
            if self.training_config.get('gradient_checkpointing', False):
                self.student_model.gradient_checkpointing_enable()
        
        # Konfiguracja LoRA
        if self.student_config.get('use_lora', True):
            print("  Konfiguracja LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.student_config['lora_r'],
                lora_alpha=self.student_config['lora_alpha'],
                lora_dropout=self.student_config['lora_dropout'],
                target_modules=self.student_config['target_modules'],
                bias="none",
            )
            
            self.student_model = get_peft_model(self.student_model, lora_config)
            self.student_model.print_trainable_parameters()
        
        print(f"✓ Model studenta załadowany")
        total_params = sum(p.numel() for p in self.student_model.parameters()) / 1e9
        trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad) / 1e6
        print(f"  Całkowite parametry: {total_params:.2f}B")
        print(f"  Trenowalne parametry: {trainable_params:.2f}M")
    
    def load_dataset(self, data_path: str = "data/teacher_dataset.jsonl") -> Dataset:
        """Ładuje dataset wygenerowany przez nauczyciela."""
        print(f"\n📚 Ładowanie datasetu z {data_path}...")
        
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                examples.append(example)
        
        print(f"✓ Załadowano {len(examples)} przykładów")
        
        # Konwersja do Dataset Hugging Face
        dataset = Dataset.from_list(examples)
        
        # Tokenizacja
        def tokenize_function(examples):
            # Łączymy prompt i response
            texts = [
                f"{prompt} {response}"
                for prompt, response in zip(examples['prompt'], examples['response'])
            ]
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Labels to input_ids (standard dla causal LM)
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizacja datasetu"
        )
        
        # Podział na train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"  Train: {len(split_dataset['train'])} przykładów")
        print(f"  Eval: {len(split_dataset['test'])} przykładów")
        
        return split_dataset
    
    def train(self, dataset_path: str = None):
        """Uruchamia proces destylacji."""
        print("\n" + "=" * 60)
        print("Rozpoczęcie treningu destylacji")
        print("=" * 60)
        
        # Użyj ścieżki z configu jeśli nie podano jawnie
        if dataset_path is None:
            data_config = self.config.get('data_generation', {})
            output_dir = data_config.get('output_dir', 'data')
            output_file = data_config.get('output_file', 'teacher_dataset.jsonl')
            dataset_path = str(Path(output_dir) / output_file)
        
        # Załaduj dataset
        dataset = self.load_dataset(dataset_path)
        
        # Konfiguracja destylacji
        distillation_config = DistillationConfig(
            kl_weight=self.training_config['kl_weight'],
            ce_weight=self.training_config['ce_weight'],
            temperature=2.0  # Wysoka temperatura dla destylacji
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config['output_dir'],
            num_train_epochs=self.training_config['num_train_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            warmup_steps=self.training_config['warmup_steps'],
            logging_steps=self.training_config['logging_steps'],
            save_steps=self.training_config['save_steps'],
            eval_steps=self.training_config['eval_steps'],
            eval_strategy="steps",  # Zmienione z evaluation_strategy
            save_strategy="steps",
            fp16=self.training_config.get('fp16', True),
            optim=self.training_config.get('optim', 'adamw_torch'),
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', True),
            max_grad_norm=self.training_config.get('max_grad_norm', 1.0),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=["wandb"] if self.config['wandb']['enabled'] else [],
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, nie masked LM
        )
        
        # Inicjalizacja custom trainera
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            distillation_config=distillation_config,
            model=self.student_model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
        )
        
        # Trening!
        print("\n🚀 Rozpoczynam trening...\n")
        trainer.train()
        
        # Zapisz finalny model
        print("\n💾 Zapisywanie finalnego modelu...")
        trainer.save_model(self.training_config['output_dir'])
        self.tokenizer.save_pretrained(self.training_config['output_dir'])
        
        print("\n" + "=" * 60)
        print("✓ Destylacja zakończona!")
        print(f"✓ Model zapisany w: {self.training_config['output_dir']}")
        print("=" * 60)


def main():
    """Główna funkcja uruchamiająca destylację."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Destylacja LLM do NPC')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Ścieżka do pliku konfiguracyjnego'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/teacher_dataset.jsonl',
        help='Ścieżka do datasetu od nauczyciela'
    )
    
    args = parser.parse_args()
    
    # Inicjalizacja i trening
    distiller = NPCDistillation(config_path=args.config)
    distiller.train(dataset_path=args.data)


if __name__ == "__main__":
    main()
