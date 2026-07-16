# Training der globalen Klassifikationsmodelle

Stand: 2026-07-16

Diese Datei bereitet das Kapitel "Durchfuehrung - Training der globalen
Klassifikationsmodelle" vor. Sie beschreibt die geplanten Trainingspipelines
fuer den globalen Bildklassifikationsvergleich. Es werden keine finalen
Trainingsergebnisse berichtet und keine langen Trainingslaeufe gestartet.

## 1. Ziel

Ziel des Trainings ist ein reproduzierbarer Vergleich globaler
Bildklassifikationsmodelle fuer vier Schleifgradklassen. Jedes Modell erhaelt
ein vollstaendiges Bild als Eingabe und sagt genau eine Klasse vorher.
Patchbasierte Heatmaps oder lokale Klassifikationskarten gehoeren nicht zu
diesem Abschnitt und werden spaeter separat behandelt.

## 2. Split-Manifest

Alle globalen Trainingslaeufe verwenden verbindlich:

```text
data/splits/bmw25_grouped_split_manifest.csv
```

Dieses Manifest enthaelt den bereits erzeugten gruppierten `70/15/15`-Split.
Es duerfen fuer die globalen Trainingslaeufe keine neuen Train/Validation/Test
Splits erzeugt werden. Die `group_id`-Zuordnung stellt sicher, dass
q1- bis q4-Ableitungen desselben Ursprungsbildes im selben Split bleiben.

## 3. Gemeinsame Trainingsvorgaben

- Rohbilder werden nicht veraendert und nicht kopiert.
- Der Dataset-Root wird nur lokal per CLI-Parameter uebergeben.
- Configs liegen unter `configs/experiments/`.
- Lokale Artefakte liegen unter `outputs/`, `runs/`, `weights/` oder
  `checkpoints/` und werden nicht committed.
- Jede Pipeline speichert lokale Run-Metadaten mit Config, Git-Commit,
  Klassenmapping, Split-Zaehlern, Seed und Paketversionen.
- Smoke-Test- oder Dry-Run-Modi pruefen Datenzugriff, Labelmapping und
  Output-Verzeichnisse, gelten aber nicht als Modelltraining.

## 4. Umgang mit Train, Validation und Test

Der Trainingssplit ist ausschliesslich fuer Parameterlernen vorgesehen. Der
Validierungssplit dient fuer Modellwahl, Hyperparameter, Checkpoint-Auswahl und
fruehe Plausibilitaetspruefungen. Das Testset bleibt bis zur finalen Evaluation
unbenutzt und darf nicht fuer Modellwahl, Patchgroessen, Schwellenwerte oder
qualitative Exploration herangezogen werden.

Checkpoint-Auswahl erfolgt ausschliesslich anhand einer Validierungsmetrik. Als
Startpunkt ist `macro_f1` vorgesehen; `balanced_accuracy` bleibt eine zentrale
ergaenzende Metrik.

## 5. Modellgruppen

### YOLOv11-cls

YOLOv11-cls dient als praxisnahe globale Klassifikationsbaseline. Der
Ultralytics-Classification-Workflow erwartet typischerweise eine lokale
Ordnerstruktur nach Split und Klasse. Da keine Bilddaten ins Repository
kopiert werden duerfen, muss diese Struktur lokal in `outputs/` oder `runs/`
vorbereitet werden, bevorzugt per Symlink- oder Link-Struktur. Echte
Modellgewichte und YOLO-Runs bleiben lokal.

### DINOv3 frozen backbone + Klassifikationskopf

DINOv3 wird als eingefrorener Feature Extractor geplant; trainiert wird nur ein
linearer oder kleiner Klassifikationskopf. DINOv3-Gewichte werden nicht
committed. Falls Gewichte nicht automatisch geladen werden sollen, muss die
lokale Gewichtsquelle vor dem echten Training dokumentiert und per Config oder
lokalem Parameter bereitgestellt werden. Feature-Caches bleiben lokal und sind
nicht Teil des Repositories.

### DeiT-Tiny from scratch

DeiT-Tiny from scratch ist die ViT-Kontrollarchitektur ohne externe
vortrainierte Gewichte. Die Config sieht `pretrained: false` vor. Dieser Ansatz
nutzt nicht dieselbe Vortrainingsinformation wie DINOv3 und dient deshalb eher
als Architekturkontrolle bzw. from-scratch-Untergrenze.

## 6. Eingabegroesse

Als konservativer gemeinsamer Startpunkt ist `224 x 224` vorgesehen. Diese
Groesse ist mit DeiT-Tiny kompatibel und erleichtert den Vergleich mit
ViT-basierten Verfahren. Fuer YOLOv11-cls kann spaeter optional `320 x 320`
validiert werden, falls Hardware und Validierungsergebnisse dafuer sprechen.

Nicht-quadratische oder abweichend grosse Bilder werden deterministisch
behandelt. Ein `resize_pad`-Ansatz ist methodisch vorsichtiger als unkritisches
Strecken, muss aber vor echten Trainingslaeufen final festgelegt werden.

## 7. Augmentierungen

Augmentierungen werden vorsichtig eingesetzt, weil Schleifstrukturen feine und
moeglicherweise richtungsabhaengige Texturinformationen enthalten. Als
Startpunkt sind leichte Helligkeits-/Kontrastaenderungen sowie Flips oder
90-Grad-Rotationen vorgesehen, sofern diese fachlich plausibel sind.

Validation und Test erhalten keine zufaelligen Augmentierungen, sondern nur
deterministisches Preprocessing.

## 8. Checkpoint-Auswahl

Checkpoints duerfen nur nach Validierungsleistung ausgewaehlt werden. Das
Testset wird nicht fuer Checkpoint-Auswahl, Early Stopping oder nachtraegliche
Schwellenwertentscheidungen genutzt. Fuer den Hauptvergleich soll die
Checkpoint-Entscheidung in der jeweiligen Run-Metadata dokumentiert werden.

## 9. Seeds

Alle Configs enthalten einen Seed. Fuer finale Vergleiche sind mehrere Seeds
wuenschenswert, falls Laufzeit und Hardwarebudget reichen. Falls nur ein Seed
genutzt wird, muss diese Einschraenkung in der Auswertung transparent genannt
werden.

## 10. Lokale Speicherorte

Vorgesehene lokale Speicherorte:

- `outputs/global_classification/<experiment_name>/` fuer Run-Metadaten,
  Predictions und kleine Auswertungsartefakte;
- `runs/` fuer Framework-spezifische Trainingslaeufe;
- `weights/` oder `checkpoints/` fuer lokale Gewichte und Checkpoints;
- `data/cache/` fuer optionale lokale Feature-Caches.

Diese Pfade sind durch `.gitignore` abgedeckt und werden nicht committed.

## 11. Versionierung

Versioniert werden:

- Experiment-Configs unter `configs/experiments/`;
- Trainings- und Dry-Run-Skripte unter `scripts/`;
- gemeinsame Daten-Utilities unter `src/`;
- methodische Dokumentation unter `docs/`;
- das bestehende Split-Manifest unter `data/splits/`.

Lokal bleiben:

- Rohdaten;
- Patches;
- Feature-Caches;
- Modellgewichte;
- Checkpoints;
- Framework-Runs;
- grosse Outputs.

## 12. Offene Punkte vor echten Trainingslaeufen

| Punkt | Status | Hinweis |
| --- | --- | --- |
| YOLOv11-Variante | offen | `yolo11n-cls` ist konservativer Startpunkt; Gewichtsquelle und Lizenzkontext muessen final dokumentiert werden. |
| DINOv3-Gewichtsquelle | offen | Lokale Bereitstellung oder erlaubter Download muss vor Training geklaert werden. |
| DINOv3-Kopfarchitektur | offen | Linearer Kopf als Startpunkt; MLP nur nach Validierungsbegruendung. |
| DeiT-Trainingsdauer | offen | From-scratch kann instabil sein; Epochen und Regularisierung nur ueber Validierung festlegen. |
| Augmentierungen | offen | Flips/Rotationen muessen fachlich plausibilisiert werden. |
| Anzahl Seeds | offen | Mehrere Seeds empfohlen, wenn Ressourcen reichen. |
| Hardwarebudget | offen | Batch Size und Laufzeit erst nach Smoke-Tests finalisieren. |
