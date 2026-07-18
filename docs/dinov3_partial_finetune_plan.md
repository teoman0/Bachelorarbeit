# DINOv3 Partial-Fine-Tuning-Plan

Stand: 2026-07-18

## Ziel

Diese Notiz beschreibt eine zusaetzliche DINOv3-Validierungsvariante fuer die
globale Klassifikation der vier Schleifgradklassen. Sie prueft, ob eine kleine
domaenenspezifische Anpassung der letzten Transformer-Bloecke die
Validierungsleistung gegenueber dem eingefrorenen DINOv3-Backbone mit
linearem Head verbessern kann.

Die Variante ist eine Optimierungsvariante und kein Ersatz fuer den
bestehenden frozen-DINOv3-Lauf. Die finale Testbewertung erfolgt spaeter
separat.

## Experiment

| Parameter | Wert |
| --- | --- |
| Experiment | `dinov3_partial_finetune_last2_bmw25_seed42` |
| Ausgangsmodell | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Head | linear, kompatibel zum bestehenden DINOv3-Head |
| Eingabegroesse | `224 x 224` |
| Batch Size | 8 |
| Batch-Fallback | 4 bei VRAM-Problemen |
| Epochen | 30 |
| Patience | 10 |
| Optimizer | AdamW |
| Backbone Learning Rate | 0.00001 |
| Head Learning Rate | 0.0005 |
| Weight Decay | 0.05 |
| Seed | 42 |
| Checkpoint-Metrik | Validation Macro-F1 |

## Trainierbare und eingefrorene Teile

Trainierbar sind:

- der lineare Klassifikationskopf;
- die letzten zwei Transformer-Bloecke des DINOv3-Backbones;
- die finale LayerNorm, falls sie in der Modellstruktur vorhanden ist.

Eingefroren bleiben:

- Patch Embedding;
- alle frueheren Transformer-Bloecke;
- alle uebrigen Backbone-Parameter.

Damit ist der Freiheitsgrad groesser als beim frozen DINOv3-Head, aber
deutlich kleiner als bei vollstaendigem Fine-Tuning.

## Split- und Testset-Regel

Verwendet wird ausschliesslich das bestehende grouped split manifest:

```text
data/splits/bmw25_grouped_split_manifest.csv
```

Es werden keine neuen Splits erzeugt. Training und Checkpoint-Auswahl nutzen
nur Train und Validation. Das Testset bleibt fuer die finale Evaluation
reserviert und wird nicht fuer Modellwahl, Hyperparameter, Early Stopping oder
qualitative Exploration verwendet.

## Ablauf

Die Pipeline liegt in:

```text
scripts/train_dinov3_partial_finetune.py
```

Der geplante lokale Ablauf ist:

1. `--dry-run`: Config, Manifest, Klassenmapping und lokale Train/Val-Dateien
   pruefen; kein Modellladen.
2. `--check-model`: DINOv3 laden, trainierbare Module markieren und
   Parameterzahlen berichten; kein Training.
3. `--smoke-test`: wenige Train/Val-Bilder laden und einen kurzen
   Forward/Backward-Test ausfuehren; keine Checkpoints.
4. `--allow-training`: spaeterer echter Partial-Fine-Tuning-Lauf nach
   erneuter expliziter Bestaetigung.

## Artefakte

Lokale Ausgaben liegen unter:

```text
outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42
```

Run-Metadaten, Predictions, Metriken, Checkpoints, Modellgewichte und andere
Outputs bleiben lokal und werden nicht committed. Versioniert werden nur
Config, Skript und Dokumentation.
