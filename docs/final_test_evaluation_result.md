# Finale Testauswertung

Stand: 2026-07-20

## Ziel

Diese Datei dokumentiert die finale Testauswertung nach abgeschlossener
Modellwahl. Der Testsplit wurde vorher nicht fuer Modellwahl,
Hyperparameterentscheidungen, qualitative Exploration oder Schwellenwertwahl
verwendet.

Die hier dokumentierten Testwerte dienen ausschliesslich der abschliessenden
Bewertung. Aus den Testwerten werden keine weiteren Modell- oder
Hyperparameterentscheidungen abgeleitet.

Grundlage war das bestehende grouped split manifest. Es wurden keine neuen
Splits erzeugt und kein Training gestartet.

## Globale Testauswertung

Alle globalen Modelle wurden auf denselben 691 Testbildern ausgewertet.

| Modell | Testfaelle | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv11n-cls | 691 Bilder | 0.9609 | 0.9490 | 0.9558 |
| DINOv3 frozen + Linear Head | 691 Bilder | 0.9392 | 0.9403 | 0.9389 |
| DeiT-Tiny from scratch | 691 Bilder | 0.6527 | 0.6236 | 0.6275 |
| DINOv3 Partial Fine-Tuning | 691 Bilder | 0.9493 | 0.9493 | 0.9510 |

### YOLOv11n-cls

| Metrik | Wert |
| --- | ---: |
| Testfaelle | 691 Bilder |
| Accuracy | 0.9609 |
| Balanced Accuracy | 0.9490 |
| Macro-F1 | 0.9558 |

Klassen-F1:

| Klasse | F1 |
| --- | ---: |
| Erste | 0.9195 |
| Final | 0.9640 |
| Fraeszustand | 0.9917 |
| Zweite | 0.9479 |

Confusion Matrix, Zeilen = True Label, Spalten = Prediction:

```text
[80, 2, 2, 8]
[1, 214, 0, 3]
[1, 0, 179, 0]
[0, 10, 0, 191]
```

### DINOv3 frozen + Linear Head

| Metrik | Wert |
| --- | ---: |
| Testfaelle | 691 Bilder |
| Accuracy | 0.9392 |
| Balanced Accuracy | 0.9403 |
| Macro-F1 | 0.9389 |

Klassen-F1:

| Klasse | F1 |
| --- | ---: |
| Erste | 0.9247 |
| Final | 0.9309 |
| Fraeszustand | 0.9944 |
| Zweite | 0.9055 |

Confusion Matrix, Zeilen = True Label, Spalten = Prediction:

```text
[86, 2, 1, 3]
[0, 202, 0, 16]
[1, 0, 179, 0]
[7, 12, 0, 182]
```

### DeiT-Tiny from scratch

| Metrik | Wert |
| --- | ---: |
| Testfaelle | 691 Bilder |
| Accuracy | 0.6527 |
| Balanced Accuracy | 0.6236 |
| Macro-F1 | 0.6275 |

Klassen-F1:

| Klasse | F1 |
| --- | ---: |
| Erste | 0.4530 |
| Final | 0.7373 |
| Fraeszustand | 0.7003 |
| Zweite | 0.6192 |

Confusion Matrix, Zeilen = True Label, Spalten = Prediction:

```text
[41, 7, 8, 36]
[9, 153, 13, 43]
[21, 11, 118, 30]
[18, 26, 18, 139]
```

### DINOv3 Partial Fine-Tuning

| Metrik | Wert |
| --- | ---: |
| Testfaelle | 691 Bilder |
| Accuracy | 0.9493 |
| Balanced Accuracy | 0.9493 |
| Macro-F1 | 0.9510 |

Klassen-F1:

| Klasse | F1 |
| --- | ---: |
| Erste | 0.9503 |
| Final | 0.9349 |
| Fraeszustand | 0.9945 |
| Zweite | 0.9242 |

Confusion Matrix, Zeilen = True Label, Spalten = Prediction:

```text
[86, 2, 2, 2]
[0, 201, 0, 17]
[0, 0, 180, 0]
[3, 9, 0, 189]
```

## Lokale Region-Modelle

Die lokale Regionenauswertung basiert auf annotierten CVAT-Bounding-Box-
Regionen. Sie ist keine semantische Segmentierung, sondern eine
rechteckbasierte lokale Klassifikation.

### DINOv3 Region-Head 4 Klassen

| Metrik | Wert |
| --- | ---: |
| Testregionen | 48 |
| `Nicht_bewertbar` ausgeschlossen | 10 |
| Accuracy | 0.7917 |
| Balanced Accuracy | 0.7925 |
| Macro-F1 | 0.7902 |

Klassen-F1:

| Klasse | F1 |
| --- | ---: |
| Erste | 0.7000 |
| Final | 0.8276 |
| Fraeszustand | 0.8333 |
| Zweite | 0.8000 |

Confusion Matrix, Zeilen = True Label, Spalten = Prediction:

```text
[7, 0, 0, 1]
[1, 12, 0, 4]
[2, 0, 5, 0]
[2, 0, 0, 14]
```

### DINOv3 Region-Head 5 Klassen

| Metrik | Wert |
| --- | ---: |
| Testregionen | 58 |
| Accuracy | 0.5862 |
| Balanced Accuracy | 0.5787 |
| Macro-F1 | 0.5722 |

Klassen-F1:

| Klasse | F1 |
| --- | ---: |
| Erste | 0.5217 |
| Zweite | 0.6061 |
| Fraeszustand | 0.6667 |
| Final | 0.6667 |
| `Nicht_bewertbar` | 0.4000 |

Confusion Matrix, Zeilen = True Label, Spalten = Prediction:

```text
[6, 1, 0, 0, 1]
[5, 10, 0, 1, 0]
[2, 0, 4, 0, 1]
[1, 5, 0, 11, 0]
[1, 1, 1, 4, 3]
```

## Interpretation

YOLOv11n-cls erreicht die hoechste globale Testleistung. DINOv3 Partial
Fine-Tuning liegt sehr nah an YOLOv11n-cls und bleibt der staerkste
transformerbasierte globale Ansatz. DINOv3 frozen zeigt weiterhin eine starke,
aber niedrigere Testleistung.

DeiT-Tiny from scratch bleibt deutlich zurueck. Das Ergebnis stuetzt die
methodische Annahme, dass vortrainierte Repraesentationen bei der vorliegenden
Datensatzgroesse und Aufgabenstellung eine zentrale Rolle spielen.

Bei den lokalen Region-Modellen ist der 4-Klassen-Region-Head der staerkste
lokale Ansatz. Der 5-Klassen-Region-Head ist schwaecher; die Sonderklasse
`Nicht_bewertbar` ist auf der kleinen Testmenge schwierig zu erkennen.

Die lokale Regionenauswertung ist als rechteckbasierte lokale Klassifikation zu
verstehen. Sie ersetzt keine semantische Segmentierung, da keine pixelgenauen
Masken ausgewertet wurden.

## Lokale Artefakte

Die folgenden lokalen Artefakte wurden fuer die finale Testauswertung erzeugt
und werden nicht committed:

- `outputs/final_test_evaluation/`
- `outputs/cvat_region_analysis/`

Predictions, Metriken, Bilder, Checkpoints und Gewichte bleiben lokale
Artefakte und sind nicht Teil des versionierten Repository-Stands.
