#!/usr/bin/env python3
"""
Szybki test GPU - sprawdza czy PyTorch widzi GPU i czy działa
"""

import torch
print("=" * 60)
print("🔍 DIAGNOSTYKA GPU")
print("=" * 60)

# 1. Czy CUDA jest dostępne?
print(f"\n1. CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ BŁĄD: PyTorch nie widzi GPU!")
    print("   Sprawdź:")
    print("   - nvidia-smi (czy GPU widoczne)")
    print("   - pip list | grep torch (czy CUDA version)")
    exit(1)

# 2. Ile GPU?
print(f"2. GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

# 3. Current device
print(f"3. Current device: {torch.cuda.current_device()}")

# 4. Memory
for i in range(torch.cuda.device_count()):
    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"4. GPU {i} Memory:")
    print(f"   Total: {total:.2f}GB")
    print(f"   Allocated: {allocated:.2f}GB")
    print(f"   Reserved: {reserved:.2f}GB")

# 5. Test obliczeń
print(f"\n5. Test obliczeń GPU...")
import time

# CPU
cpu_tensor = torch.randn(1000, 1000)
start = time.time()
cpu_result = cpu_tensor @ cpu_tensor
cpu_time = time.time() - start
print(f"   CPU: {cpu_time:.4f}s")

# GPU
gpu_tensor = torch.randn(1000, 1000).cuda()
torch.cuda.synchronize()
start = time.time()
gpu_result = gpu_tensor @ gpu_tensor
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"   GPU: {gpu_time:.4f}s")
print(f"   Speedup: {cpu_time/gpu_time:.1f}x")

if gpu_time > cpu_time:
    print(f"\n⚠️  UWAGA: GPU wolniejsze niż CPU!")
    print(f"   To NIE JEST NORMALNE - sprawdź instalację!")

# 6. Test ładowania modelu
print(f"\n6. Test ładowania małego modelu...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # Tiny model for test
print(f"   Ładowanie {model_name}...")

try:
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
    )
    load_time = time.time() - start
    
    print(f"   ✓ Załadowano w {load_time:.2f}s")
    print(f"   Device: {next(model.parameters()).device}")
    
    # Test generacji
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer("Hello world", return_tensors="pt").to("cuda:0")
    
    print(f"\n   Test generacji (10 tokenów)...")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    gen_time = time.time() - start
    
    print(f"   ✓ Wygenerowano w {gen_time:.3f}s")
    print(f"   Tekst: {tokenizer.decode(outputs[0])}")
    
    if gen_time > 5.0:
        print(f"\n   ⚠️  WOLNO! (>5s dla 10 tokenów)")
        print(f"   Coś jest nie tak z GPU!")
    else:
        print(f"\n   ✓ Szybkość OK!")
        
except Exception as e:
    print(f"   ❌ Błąd: {e}")

print("\n" + "=" * 60)
print("PODSUMOWANIE:")
print("=" * 60)
if torch.cuda.is_available() and gpu_time < cpu_time * 5:
    print("✅ GPU działa poprawnie")
else:
    print("❌ Problem z GPU - sprawdź instalację PyTorch")
print("=" * 60)
