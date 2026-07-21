# Methodik

## Zweck und Geltungsbereich

Dieses Dokument ist die kanonische Beschreibung des Untersuchungsdesigns, der
Modellvergleichslogik sowie der Trennung von Training, Validation und finaler
Testbewertung. Es beschreibt globale und lokale Aufgaben auf methodischer
Ebene; konkrete Parameter und Ergebnisse werden in den jeweils verlinkten
Dokumenten geführt.

## Forschungsziel

Ziel der Untersuchung ist die bildbasierte Klassifikation des Schleif- und
Bearbeitungszustands metallischer Oberflächen. Im Mittelpunkt steht die Frage,
wie sich eine praxisnahe vortrainierte Klassifikationsbaseline, stark
vortrainierte Vision-Transformer-Repräsentationen und ein Vision Transformer
ohne externe Vortrainingsinformation auf derselben Datenbasis verhalten.

Die Untersuchung umfasst zwei getrennte Aufgabenebenen:

1. globale Bildklassifikation mit genau einem Schleifgradlabel pro Bild;
2. lokale Klassifikation manuell markierter rechteckiger Bildregionen.

Die lokale Analyse ist keine semantische Segmentierung. Die CVAT-Annotationen
sind Bounding Boxes und keine pixelgenauen Masken. Entsprechend werden weder
Pixel-Accuracy noch IoU oder andere Segmentierungsmetriken berichtet.

## Modellgruppen

Der globale Vergleich umfasst:

- `YOLOv11n-cls` als praxisnahe vortrainierte Klassifikationsbaseline;
- DINOv3 ViT-B/16 mit eingefrorenem Backbone und linearem Head;
- `deit_tiny_patch16_224` mit `pretrained=false` als ViT-from-scratch-
  Kontrollmodell;
- DINOv3 ViT-B/16 mit partiellem Fine-Tuning der letzten zwei
  Transformer-Blöcke, der finalen Norm und des linearen Heads.

Das Vortraining ist ein zentraler methodischer Unterschied. YOLOv11n-cls und
DINOv3 beginnen mit extern gelernten Repräsentationen. DeiT-Tiny wird ohne
vortrainierte Gewichte vollständig auf dem vorliegenden Datensatz optimiert.
Seine Rolle ist daher eine Architekturkontrolle und nicht die eines in Bezug
auf Vortrainingsinformation identischen Vergleichs.

Die lokale Analyse verwendet zunächst den globalen DINOv3-Partial-Fine-
Tuning-Checkpoint direkt auf Region-Crops. Anschließend werden bei
eingefrorenem DINOv3-Backbone kleine MLP-Heads für vier Klassen sowie für vier
Schleifgradklassen plus `Nicht_bewertbar` trainiert.

## Datensplit und Leakage-Schutz

Alle Aufgaben verwenden das bestehende gruppierte Split-Manifest. Die
Gruppierung entfernt q1- bis q4-Suffixe, sodass Viertel desselben
Ursprungsbildes immer im gleichen Split bleiben. Der 70/15/15-Split umfasst
3.225 Trainings-, 691 Validierungs- und 691 Testbilder. Er wurde vor jeder
Patch-, Crop- oder Regionenverarbeitung festgelegt.

Training dient ausschließlich der Parameteroptimierung. Validation wird für
Checkpoint-Auswahl, Architekturvarianten, Crop-Strategien und
Modellentscheidungen genutzt. Nach Abschluss dieser Entscheidungen erfolgt
die finale Testauswertung ohne nachträgliche Anpassung von Modellen,
Hyperparametern oder Schwellenwerten.

Der Testsplit wurde nicht für Training, Checkpoint-Auswahl,
Hyperparameter-Tuning oder Leistungsbewertung während der Modellentwicklung
verwendet. Technische Integritätsprüfungen erfassten jedoch Dateiexistenz und
Ordnerstruktur; ein einzelnes Testbild wurde in einem Smoke-Test dekodiert.
Vor der finalen Evaluation erfolgten keine Forward Passes, Testprädiktionen
oder Testmetriken.

## Bewertungskriterien

Für globale und rechteckbasierte Klassifikation werden dieselben grundlegenden
Metriken verwendet:

| Metrik | Bedeutung |
| --- | --- |
| Accuracy | Anteil korrekt klassifizierter Beispiele über alle Klassen |
| Balanced Accuracy | Mittelwert der klassenweisen Recall-Werte |
| Macro-F1 | Ungewichteter Mittelwert der klassenweisen F1-Werte |
| Precision je Klasse | Anteil korrekter Beispiele unter allen Vorhersagen einer Klasse |
| Recall je Klasse | Anteil erkannter Beispiele unter allen tatsächlichen Beispielen einer Klasse |
| F1 je Klasse | Harmonisches Mittel aus Precision und Recall |
| Confusion Matrix | Häufigkeiten wahrer und vorhergesagter Klassen |

Aufgrund der ungleichen Klassenverteilung reicht Accuracy allein nicht aus.
Macro-F1 war die primäre Validierungsmetrik für die projektseitige
Checkpoint- und Variantenwahl; Balanced Accuracy ergänzt die Bewertung der
klassenweisen Trefferquoten. Framework-interne Checkpoint-Logik, etwa bei
Ultralytics, wird davon getrennt dokumentiert.

## Globale Bildklassifikation

Jedes globale Modell erhält ein vollständiges Bild und gibt genau eine der
vier Klassen aus. Modellabhängige Augmentierungen werden nur im Training
angewendet; Validation und Test verwenden deterministisches Preprocessing.
Klassenreihenfolge, Split-Manifest, Seed, Config, Git-Commit und
Paketversionen werden in lokalen Metadaten festgehalten.

Das DINOv3-Preprocessing entspricht der Trainingspipeline:

```text
EXIF-Transpose
→ RGB
→ seitenverhältnistreues BICUBIC-Resize
→ schwarzes Padding auf 224 × 224
→ DINOv3-Processor
```

Bei einer ersten finalen DINOv3-Auswertung war die Bildvorbereitung nicht
vollständig konsistent mit dem Training. Die beiden DINOv3-Testläufe wurden
deshalb mit unveränderten Checkpoints, unverändertem Testmanifest und der
kanonischen Pipeline technisch korrigiert. Es erfolgten weder eine neue
Modellwahl noch Hyperparameter- oder Schwellenwertänderungen. Die korrigierten
Ergebnisse sind in `docs/final_test_evaluation_result.md` dokumentiert.

## Rechteckbasierte lokale Klassifikation

Die lokalen Regionen stammen aus manuellen CVAT-Rechtecken. Bounding Boxes
werden an Bildgrenzen geclippt und entweder quadratisch aufgefüllt oder direkt
auf 224 × 224 Pixel skaliert. Crop-Strategien wurden ausschließlich auf
Validation verglichen. `Nicht_bewertbar` wird in der Vier-Klassen-Auswertung
ausgeschlossen und nur im separaten Fünf-Klassen-Experiment als Zielklasse
verwendet.

Der lokale Head nutzt eingefrorene DINOv3-Features. Dadurch wird nur die
Zuordnung der vorhandenen Repräsentation zu lokalen Regionlabels gelernt. Die
Ergebnisse dienen der räumlichen Plausibilisierung der globalen Modelle, sind
aber wegen der kleinen Annotationsbasis, des Kontextverlusts beim Cropping und
der rechteckigen Annotationen nicht als pixelgenaue Oberflächenanalyse zu
interpretieren.

## Reproduzierbarkeit und Ergebnisgrenzen

Experiment-Configs, Trainings- und Evaluationsskripte sowie das Split-Manifest
sind versioniert. Rohdaten, lokale Dataset-Strukturen, Feature-Caches,
Predictions, Metrikartefakte, Runs und Checkpoints bleiben außerhalb von Git.
Die finale Testauswertung ist durch das verpflichtende Flag
`--allow-final-test` gesperrt. Testwerte werden ausschließlich als finale
Bewertung berichtet und dürfen keine weiteren Modellentscheidungen auslösen.

## Verwandte Dokumente

Datensatz und Split stehen in der [Datensatzbeschreibung](dataset.md). Die
tatsächlich ausgeführten Experimente beschreibt der [experimentelle
Aufbau](experimental_setup.md), ihre Modellauswahl die
[Validierungsauswertung](validation_results.md) und die lokale Untersuchung
die [Regionenanalyse](region_analysis.md). Die abschließenden Testwerte bleiben
in der [finalen Testauswertung](final_test_evaluation_result.md) kanonisch
dokumentiert.
