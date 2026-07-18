# DINOv3 Partial-Fine-Tuning-Ergebnis

Stand: 2026-07-18

## Ziel

Diese Datei dokumentiert den abgeschlossenen DINOv3-Partial-Fine-Tuning-Lauf
fuer die globale Klassifikation der vier Schleifgradklassen. Die Zahlen sind
Validierungsergebnisse und keine finalen Testwerte.

Die Variante dient als DINOv3-Optimierungsvariante mit begrenzter
domaenenspezifischer Anpassung. Sie ersetzt nicht den frozen-DINOv3-Ansatz,
sondern prueft, ob die letzten Transformer-Bloecke auf dem Validierungssplit
einen Mehrwert bringen.

## Experiment

| Parameter | Wert |
| --- | --- |
| Name | `dinov3_partial_finetune_last2_bmw25_seed42` |
| Modell | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Feature-Dimension | 768 |
| Rolle | DINOv3-Optimierungsvariante mit begrenzter domaenenspezifischer Anpassung |
| Split | bestehendes grouped split manifest |
| Train | 3225 Bilder |
| Validation | 691 Bilder |
| Test | nicht verwendet |

Das Testset blieb unberuehrt und wird erst fuer die finale Evaluation genutzt.

## Trainierbare Module

Trainierbar waren:

- `model.layer.10`;
- `model.layer.11`;
- finale `norm`;
- linearer Head.

Eingefroren blieben Patch Embedding und die frueheren Transformer-Bloecke.

## Parameter

| Parametergruppe | Anzahl |
| --- | ---: |
| Gesamtparameter inkl. Head | 85,663,492 |
| Trainierbare Parameter inkl. Head | 14,181,892 |

## Training

| Parameter | Wert |
| --- | --- |
| Image Size | 224 |
| Batch Size | 8 |
| Epochen geplant | 30 |
| Tatsaechlich gelaufene Epochen | 14 |
| Early Stopping | ja |
| Patience | 10 |
| Bester Epoch | 4 nach Validation Macro-F1 |
| Laufzeit | ca. 37 min 39 s |
| Optimizer | AdamW |
| Backbone Learning Rate | 0.00001 |
| Head Learning Rate | 0.0005 |
| Weight Decay | 0.05 |
| Seed | 42 |

Checkpoint-Auswahl erfolgte ausschliesslich anhand von Validation Macro-F1.

## Gesamtmetriken

| Modell | Accuracy (Val) | Balanced Accuracy (Val) | Macro-F1 (Val) |
| --- | ---: | ---: | ---: |
| DINOv3 Partial Fine-Tuning last2 | 0.9681620839 | 0.9662386859 | 0.9696503456 |

## Klassenweise Metriken

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe Viertel | 1.000000 | 0.945652 | 0.972067 |
| Finaler Zustand Viertel | 0.971963 | 0.954128 | 0.962963 |
| Fräszustand Viertel | 0.994475 | 1.000000 | 0.997230 |
| Zweite Bearbeitungsstufe Viertel | 0.928230 | 0.965174 | 0.946341 |

## Confusion Matrix

Zeilen sind wahre Klassen, Spalten sind vorhergesagte Klassen.

| True \ Pred | Erste | Final | Fräszustand | Zweite |
| --- | ---: | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 87 | 0 | 0 | 5 |
| Finaler Zustand | 0 | 208 | 0 | 10 |
| Fräszustand | 0 | 0 | 180 | 0 |
| Zweite Bearbeitungsstufe | 0 | 6 | 1 | 194 |

## Vergleich zu Frozen DINOv3

| Variante | Accuracy (Val) | Balanced Accuracy (Val) | Macro-F1 (Val) |
| --- | ---: | ---: | ---: |
| Frozen DINOv3 + linear head | 0.9551374819 | 0.9517583436 | 0.9555180393 |
| Partial Fine-Tuning last2 | 0.9681620839 | 0.9662386859 | 0.9696503456 |

Die Verbesserung im Macro-F1 betraegt ca. `+0.0141`.

## Interpretation

Das partielle Fine-Tuning verbessert die Validierungsleistung gegenueber dem
eingefrorenen DINOv3-Backbone. Die Verbesserung betrifft insbesondere die
schwierigen Uebergaenge zwischen den Bearbeitungsstufen. Die Klasse
`Fräszustand Viertel` wird im Validierungssplit vollstaendig korrekt erkannt.

Das Experiment zeigt, dass die vortrainierten DINOv3-Repraesentationen bereits
stark sind, durch eine begrenzte Anpassung der letzten Transformer-Bloecke aber
weiter an die Domaene metallischer Oberflaechen angepasst werden koennen.

Diese Ergebnisse sind Validierungsergebnisse. Das Testset bleibt fuer die
finale Evaluation reserviert.

## Lokale Artefakte

Die folgenden Dateien wurden lokal erzeugt und werden nicht committed:

- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/run_metadata.json`;
- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/training_log.csv`;
- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/training_metrics.csv`;
- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/checkpoints/best_model.pt`;
- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/checkpoints/last_model.pt`;
- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/predictions_val.csv`;
- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/val_metrics.json`;
- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/val_metrics.csv`;
- `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/confusion_matrix_val.csv`.
