# Finale Testauswertung

Stand: 2026-07-21

## Ziel

Diese Datei dokumentiert die finale Testauswertung nach abgeschlossener
Modellwahl. Der Testsplit wurde waehrend der Modellentwicklung nicht fuer
Training, Checkpoint-Auswahl, Hyperparameter-Tuning oder Leistungsbewertung
verwendet. Technische Integritaetspruefungen erfassten jedoch Dateiexistenz
und Ordnerstruktur. Ein einzelnes Testbild wurde im Rahmen eines Smoke-Tests
dekodiert. Vor der finalen Evaluation erfolgten keine Forward Passes,
Testpraediktionen oder Testmetriken.

Die hier dokumentierten Testwerte dienen ausschliesslich der abschliessenden
Bewertung. Aus den Testwerten werden keine weiteren Modell- oder
Hyperparameterentscheidungen abgeleitet.

Grundlage war das bestehende grouped split manifest. Es wurden keine neuen
Splits erzeugt und kein Training gestartet.

## Technisch korrigierte DINOv3-Auswertung

Die hier dokumentierte korrigierte DINOv3-Auswertung ersetzt die fruehere
DINOv3-Testauswertung. Grund war eine Inkonsistenz zwischen dem beim Training
und dem bei der ersten finalen Evaluation verwendeten Bild-Preprocessing.
Die technische Korrektur verwendet nun fuer beide globalen DINOv3-Modelle
exakt die Trainingspipeline:

```text
EXIF-Transpose
-> RGB
-> seitenverhaeltnistreues BICUBIC-Resize
-> schwarzes Padding auf 224 x 224
-> DINOv3-Processor
```

Die korrigierte Auswertung wurde mit Git-Commit
`f6d56feb0994554b2acd6cba2a06358594d9d1db` ausgefuehrt. Checkpoints,
Klassenreihenfolge und das gruppierte Testmanifest mit 691 Bildern blieben
unveraendert. Es wurden weder Modelle, Hyperparameter, Schwellenwerte noch
Checkpoints angepasst. Die erneute Auswertung war ausschliesslich eine
technische Fehlerkorrektur und fuehrte zu keiner neuen Modellentscheidung.

Bei DINOv3 frozen aenderten sich 6 von 691 vorhergesagten Klassen, beim
Partial Fine-Tuning 4 von 691. Die Accuracy blieb bei beiden Modellen
unveraendert. Unter den 22 nichtquadratischen Testbildern aenderte sich keine
vorhergesagte Klasse. Die Modellrangfolge blieb ebenfalls unveraendert.

## Globale Testauswertung

Alle globalen Modelle wurden auf denselben 691 Testbildern ausgewertet.

| Modell | Testfaelle | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv11n-cls | 691 Bilder | 0.9609 | 0.9490 | 0.9558 |
| DINOv3 frozen + Linear Head | 691 Bilder | 0.9392185239 | 0.9405202828 | 0.9396630903 |
| DeiT-Tiny from scratch | 691 Bilder | 0.6527 | 0.6236 | 0.6275 |
| DINOv3 Partial Fine-Tuning | 691 Bilder | 0.9493487699 | 0.9508454685 | 0.9517624382 |

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
| Accuracy | 0.9392185239 |
| Balanced Accuracy | 0.9405202828 |
| Macro-F1 | 0.9396630903 |

Klassen-F1:

| Klasse | F1 |
| --- | ---: |
| Erste | 0.929730 |
| Final | 0.928074 |
| Fraeszustand | 0.994444 |
| Zweite | 0.906404 |

Confusion Matrix, Zeilen = True Label, Spalten = Prediction:

```text
[86, 2, 1, 3]
[0, 200, 0, 18]
[1, 0, 179, 0]
[6, 11, 0, 184]
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

Zum Trainingsverlauf sind 92 Epochen protokolliert. Der beste Checkpoint
stammt aus Epoche 68; danach sind 24 Epochen ohne weitere Verbesserung
protokolliert. Ein regulaerer Abschluss durch Early Stopping ist jedoch nicht
vollstaendig dokumentiert und wird daher nicht als nachweislich ausgeloest
angegeben.

### DINOv3 Partial Fine-Tuning

| Metrik | Wert |
| --- | ---: |
| Testfaelle | 691 Bilder |
| Accuracy | 0.9493487699 |
| Balanced Accuracy | 0.9508454685 |
| Macro-F1 | 0.9517624382 |

Klassen-F1:

| Klasse | F1 |
| --- | ---: |
| Erste | 0.956044 |
| Final | 0.934579 |
| Fraeszustand | 0.994475 |
| Zweite | 0.921951 |

Confusion Matrix, Zeilen = True Label, Spalten = Prediction:

```text
[87, 1, 2, 2]
[0, 200, 0, 18]
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

YOLOv11n-cls erreicht weiterhin den hoechsten globalen Macro-F1. DINOv3
Partial Fine-Tuning erreicht die hoechste Balanced Accuracy und bleibt der
staerkste transformerbasierte globale Ansatz. DINOv3 frozen zeigt weiterhin
eine starke, aber niedrigere Testleistung. Die technisch korrigierten Werte
aendern die Modellrangfolge nicht.

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
- `outputs/final_test_evaluation_reproducible/`
- `outputs/cvat_region_analysis/`

Predictions, Metriken, Bilder, Checkpoints und Gewichte bleiben lokale
Artefakte und sind nicht Teil des versionierten Repository-Stands.
