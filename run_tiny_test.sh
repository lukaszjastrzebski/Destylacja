#!/bin/bash

# TEST PIPELINE - Małe modele dla PC
# Teacher: TinyLlama-1.1B (~2.5GB VRAM)
# Student: Qwen2-0.5B (~1GB VRAM)
# Razem: ~4-5GB VRAM

echo "=========================================="
echo "🧪 TEST: TinyLlama-1.1B → Qwen2-0.5B"
echo "=========================================="
echo ""
echo "Nauczyciel: TinyLlama-1.1B-Chat"
echo "Student: Qwen2-0.5B-Instruct"
echo "Samples: 2878 (wszystkie z Excel)"
echo "VRAM: ~4-5GB (działa na słabszym PC)"
echo ""
echo "📁 Pliki wyjściowe:"
echo "   Dataset: data/teacher_dataset.jsonl"
echo "   Model: models/tinyllama_to_qwen_npc/"
echo ""

# Kolory
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Sprawdź venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment nie jest aktywowane!${NC}"
    echo "Uruchom: source .venv/bin/activate"
    exit 1
fi

# Sprawdź Excel
if [ ! -f "prompty.xlsx" ]; then
    echo -e "${RED}❌ Brak pliku prompty.xlsx!${NC}"
    exit 1
fi

# Info o GPU (opcjonalne)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}Sprawdzam GPU...${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

read -p "Kontynuować? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Anulowano."
    exit 0
fi

echo ""
echo -e "${BLUE}[1/3] Generowanie danych od nauczyciela (1-3h)...${NC}"
echo "TinyLlama-1.1B będzie generować odpowiedzi na 2878 promptów z Excel"
echo ""

python scripts/generate_teacher_data.py --config config_tiny_test.yaml

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Błąd podczas generowania danych!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dane wygenerowane${NC}\n"

echo -e "${BLUE}[2/3] Trening destylacji (1-2h)...${NC}"
echo "Student Qwen2-0.5B uczy się od nauczyciela TinyLlama-1.1B"
echo ""

python scripts/train_distillation.py --config config_tiny_test.yaml

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Błąd podczas treningu!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Destylacja zakończona${NC}\n"

echo -e "${BLUE}[3/3] Ewaluacja modelu...${NC}"
python scripts/evaluate_model.py \
    --student_path models/tinyllama_to_qwen_npc \
    --config config_tiny_test.yaml \
    --teacher_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --mode compare

echo ""
echo "=========================================="
echo -e "${GREEN}🏆 TEST ZAKOŃCZONY! 🏆${NC}"
echo "=========================================="
echo ""
echo "📁 Pliki:"
echo "   Model: models/tinyllama_to_qwen_npc/"
echo "   Dataset: data/teacher_dataset.jsonl"
echo ""
echo "📊 Rozmiar modelu:"
ls -lh models/tinyllama_to_qwen_npc/ 2>/dev/null | tail -n +2
echo ""
echo "🎮 Następne kroki:"
echo "  1. Test interaktywny:"
echo "     python scripts/evaluate_model.py \\"
echo "       --student_path models/tinyllama_to_qwen_npc \\"
echo "       --config config_tiny_test.yaml \\"
echo "       --mode interactive"
echo ""
echo "  2. Spakuj model:"
echo "     tar -czf tiny_npc_model.tar.gz models/tinyllama_to_qwen_npc/"
echo ""
echo "✨ Jeśli test działa OK, możesz użyć większych modeli na serwerze!"
echo ""
