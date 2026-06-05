# GTCRN_BC

GTCRN, который использует BC (bone conduction) модальность.

В обучение жестко зашиты датасеты:
1. `verbreb/demand_noise_subset_16k` - поднабор из 30 семплов [voice-biomarkers/DEMAND-acoustic-noise](https://huggingface.co/datasets/voice-biomarkers/DEMAND-acoustic-noise). Обрезаны до 20 сек., 16 kHz.
2. `verbreb/vibravox_16k_8s_headset_temple_full` - поднабор с vibravox. Обрезаны до 8 сек, 16 kHz. Только 2 колонки:
    1. `headset_microphone`
    2. `temple_vibration_pickup`

```bash
# тестовый запуск, проверка, что основные функции работают
python3 -m train --check
```
### 05.05.2026
Зашил датасеты с файлами длиной до 8 сек.
Так как далеко не все имеют длину 8 сек., - пришлось делать padding.
В лоссе padding не учитывается ни для звукового сигнала ни для его спектрограммы (через маску).