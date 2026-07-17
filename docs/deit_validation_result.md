# DeiT-Tiny-Validierungsergebnis

Stand: 2026-07-17

## Ziel

Diese Datei dokumentiert den finalisierten Validierungsstand des
DeiT-Tiny-from-scratch-Laufs fuer die globale Klassifikation der vier
Schleifgradklassen. Die Zahlen sind Validierungsergebnisse und keine finalen
Testwerte.

Der Lauf dient als ViT-from-scratch-Kontrollmodell ohne externe
Vortrainingsinformation. Er ist daher nicht direkt mit Verfahren vergleichbar,
die vortrainierte Repraesentationen oder Gewichte nutzen.

## Modell und Split

| Parameter | Wert |
| --- | --- |
| Modell | `deit_tiny_patch16_224` |
| Quelle | `timm` |
| Pretrained | `false` |
| Rolle | ViT-from-scratch-Kontrollmodell |
| Split | bestehendes grouped split manifest |
| Train | 3225 Bilder |
| Validation | 691 Bilder |
| Test | nicht verwendet |

Das Testset blieb unberuehrt und wird erst fuer die finale Evaluation genutzt.

## Trainings- und Auswertungsstand

| Punkt | Wert |
| --- | --- |
| Protokollierte Epochen | 92 |
| Bester Epoch | 68 nach Validation Macro-F1 |
| Checkpoint-Metrik | Validation Macro-F1 |
| Auswertung | gespeicherter bester Checkpoint |
| Bester Checkpoint | `outputs/global_classification/deit_tiny_scratch_bmw25_seed42/checkpoints/best_model.pt` |
| Validierungspraediktionen | `outputs/global_classification/deit_tiny_scratch_bmw25_seed42/predictions_val.csv` |

Der Lauf wurde aus den vorhandenen lokalen Artefakten finalisiert. Das
stdout-Log enthielt 92 protokollierte Epochen. Der beste Checkpoint bestaetigt
intern Epoch 68 und dieselben Validierungsmetriken wie die lokale
Metrikdatei. Ein sauberer Abschlussmarker war nicht vorhanden; `train_stderr`
war leer.

## Gesamtmetriken

| Modell | Accuracy (Val) | Balanced Accuracy (Val) | Macro-F1 (Val) |
| --- | ---: | ---: | ---: |
| DeiT-Tiny from scratch | 0.6801736614 | 0.6434774009 | 0.6476802764 |

## Klassenweise Metriken

| Klasse | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Erste Bearbeitungsstufe Viertel | 0.476190 | 0.434783 | 0.454545 |
| Finaler Zustand Viertel | 0.776786 | 0.798165 | 0.787330 |
| Fräszustand Viertel | 0.778523 | 0.644444 | 0.705167 |
| Zweite Bearbeitungsstufe Viertel | 0.598291 | 0.696517 | 0.643678 |

## Confusion Matrix

Zeilen sind wahre Klassen, Spalten sind vorhergesagte Klassen.

| True \ Pred | Erste | Final | Fräszustand | Zweite |
| --- | ---: | ---: | ---: | ---: |
| Erste Bearbeitungsstufe | 40 | 7 | 9 | 36 |
| Finaler Zustand | 6 | 174 | 11 | 27 |
| Fräszustand | 19 | 14 | 116 | 31 |
| Zweite Bearbeitungsstufe | 19 | 29 | 13 | 140 |

## Interpretation

DeiT-Tiny from scratch bleibt deutlich hinter den vortrainierten Ansaetzen
YOLOv11n-cls und DINOv3 zurueck. Besonders schwach ist die Klasse
`Erste Bearbeitungsstufe Viertel`, bei der Precision, Recall und F1 klar
niedriger ausfallen als bei den anderen Modellfamilien.

Das Ergebnis stuetzt die Annahme, dass ein Vision Transformer ohne
Vortraining bei dem vorliegenden Datensatz deutlich schwerer zu trainieren
ist. Der Lauf wird deshalb als Architekturkontrolle ohne externe
Vortrainingsinformation eingeordnet, nicht als primaerer Kandidat fuer die
spaetere finale Testbewertung.

## Testset-Hinweis

Die hier dokumentierten Werte stammen ausschliesslich aus dem
Validierungssplit. Das Testset wurde nicht fuer Training, Early Stopping,
Checkpoint-Auswahl, Hyperparameter- oder Modellentscheidung genutzt. Die
finale Testbewertung erfolgt spaeter separat.
