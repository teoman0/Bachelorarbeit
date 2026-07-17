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

Fuer YOLOv11-cls wird die lokale Classification-Struktur direkt aus dem
versionierten Split-Manifest erzeugt:

```text
outputs/global_classification/yolov11_cls/yolo_dataset/
  train/<klasse>/
  val/<klasse>/
  test/<klasse>/
```

Das Skript nutzt keine Bildkopien. Standardmaessig wird zuerst ein Symlink
versucht; falls das unter Windows wegen Rechten nicht funktioniert, wird ein
Hardlink angelegt. Die lokale Summary dokumentiert Klassenmapping,
Split-Zaehler und verwendete Link-Methoden. Die Test-Ordner werden nur fuer
die spaetere finale Evaluation materialisiert und duerfen nicht fuer Training,
Validation, Early Stopping oder Modellwahl verwendet werden.

Der Hardware-Startpunkt ist ein lokales System mit NVIDIA GeForce RTX 4060 Ti
und 8 GB dediziertem VRAM. Die geplante YOLO-Startkonfiguration lautet:

| Parameter | Startwert |
| --- | --- |
| Modell | `yolo11n-cls` |
| optionale Variante | `yolo11s-cls` nur nach erfolgreichem n-Modell und nur anhand Validation |
| Bildgroesse | `320` |
| Fallback-Bildgroesse | `224` |
| Batch Size | `16` |
| Epochen | `75` |
| Patience | `15` |
| AMP / mixed precision | `true` |
| Device | `0` |
| Workers | `4` |

Ultralytics nutzt fuer Classification-Training intern eigene
Validierungsmetriken, typischerweise Top-1-Accuracy oder Validation Loss, fuer
Checkpointing und Fortschritt. Die in dieser Arbeit bevorzugte Macro-F1 wird
deshalb separat ueber den eigenen Evaluationsworkflow auf Prediction-Dateien
berechnet und nicht als automatisch verfuegbare Ultralytics-Checkpointmetrik
vorausgesetzt.

### DINOv3 frozen backbone + Klassifikationskopf

DINOv3 wird als eingefrorener Feature Extractor geplant; trainiert wird nur ein
linearer Klassifikationskopf. Primaerer Startpunkt ist
`facebook/dinov3-vitb16-pretrain-lvd1689m` mit `224 x 224` Eingabegroesse,
Batch Size `16`, AdamW, `learning_rate=0.001`, `weight_decay=0.0001` und
`seed=42`. ViT-B/16 ist der Kompromiss zwischen Modellstaerke und lokalem
Hardwarebedarf. ViT-L/16 wird fuer den ersten Lauf nicht verwendet, weil 8 GB
VRAM knapp werden koennen; `facebook/dinov3-vits16-pretrain-lvd1689m` bleibt
der Fallback, falls ViT-B/16 nicht sauber laedt oder laeuft.

Der Backbone bleibt eingefroren, weil die Arbeit die Qualitaet der
DINOv3-Repraesentationen fuer die Schleifgradbewertung untersuchen soll. Der
lineare Head ist der erste saubere Ansatz; ein MLP-Head kann spaeter nur als
separate Validierungsvariante betrachtet werden. Checkpoint-Auswahl erfolgt
ueber Validation Macro-F1.

DINOv3-Gewichte werden nicht committed. Der Zugriff erfolgt ueber Hugging Face
oder eine lokal bereitgestellte Gewichtsquelle. `allow_download` ist in der
Config standardmaessig `false`; Downloads sind nur mit explizitem
CLI-Override erlaubt. Hugging-Face-Tokens oder andere Credentials duerfen nie
in das Repository geschrieben werden. Feature-Caches, Head-Checkpoints,
Predictions und Metriken bleiben lokal und ignored.

Der geplante Ablauf ist:

1. `--dry-run`: Config, Manifest, Klassenmapping und lokale Train/Val-Dateien
   pruefen; kein Modell laden.
2. `--check-model`: DINOv3 nur lokal laden oder mit explizitem
   `--allow-download`; Feature-Dimension und Repraesentation pruefen; kein
   Training.
3. `--smoke-test`: sehr wenige Train/Val-Bilder laden, Features erzeugen und
   einen kurzen Forward/Backward-Test des linearen Heads ausfuehren; keine
   Ergebnisinterpretation.
4. `--allow-training`: spaeterer echter Head-Trainingslauf nach erneuter
   expliziter Bestaetigung; nutzt Train und Validation, nicht Test.

Der detaillierte Plan liegt in [dinov3_training_plan.md](dinov3_training_plan.md).
Das erste DINOv3-Validierungsergebnis ist in
[dinov3_validation_result.md](dinov3_validation_result.md) dokumentiert.

### DeiT-Tiny from scratch

DeiT-Tiny from scratch ist die ViT-Kontrollarchitektur ohne externe
vortrainierte Gewichte. Verwendet wird `deit_tiny_patch16_224` aus `timm` mit
`pretrained=false`, `224 x 224` Eingabegroesse, Batch Size `16`, AdamW,
`learning_rate=0.0005`, `weight_decay=0.05`, `epochs=150` und `seed=42`.
Dieser Ansatz nutzt nicht dieselbe Vortrainingsinformation wie DINOv3 und ist
deshalb eine Architekturkontrolle bzw. from-scratch-Untergrenze.

Weil der Datensatz fuer ein Transformer-Modell ohne Vortraining klein ist, ist
dieser Ansatz erwartbar schwieriger und potenziell instabiler als DINOv3 mit
vortrainiertem frozen Backbone. Es werden keine pretrained weights geladen
oder committed. Checkpoint-Auswahl erfolgt ausschliesslich ueber Validation
Macro-F1; das Testset bleibt bis zur finalen Evaluation unberuehrt.

Der geplante Ablauf ist:

1. `--dry-run`: Config, Manifest, Klassenmapping und lokale Train/Val-Dateien
   pruefen; kein Modelltraining.
2. `--check-model`: `timm.create_model(..., pretrained=False, num_classes=4)`
   ausfuehren und Parameterzahl/Device pruefen; kein Training.
3. `--smoke-test`: wenige Train/Val-Bilder laden und einen kurzen
   Forward/Backward-Test ausfuehren; keine Ergebnisinterpretation.
4. `--allow-training`: spaeterer echter from-scratch-Lauf nach erneuter
   expliziter Bestaetigung; nutzt Train und Validation, nicht Test.

Der detaillierte Plan liegt in [deit_training_plan.md](deit_training_plan.md).

## 6. Eingabegroesse

Als konservativer gemeinsamer ViT-Startpunkt ist `224 x 224` vorgesehen. Diese
Groesse ist mit DeiT-Tiny kompatibel und erleichtert den Vergleich mit
ViT-basierten Verfahren. Fuer YOLOv11-cls wird aufgrund der lokalen RTX 4060 Ti
mit 8 GB VRAM ein aussagekraeftiger Startlauf mit `320 x 320` und Batch Size
`16` vorbereitet. `224 x 224` bleibt der Fallback, falls Speicher- oder
Laufzeitprobleme auftreten.

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

## 10.1 YOLO-Befehle

Die folgenden Befehle nutzen den lokalen Dataset-Root nur als CLI-Parameter.
Dieser Pfad wird nicht in versionierte Dateien geschrieben.

Dry-Run mit Manifest-, Klassen- und Dateipruefung:

```powershell
python scripts/train_yolov11_cls.py `
  --dataset-root <lokaler_dataset_root> `
  --dry-run `
  --smoke-test `
  --max-smoke-samples 2
```

Lokale YOLO-Dataset-Struktur aus dem Split-Manifest erzeugen:

```powershell
python scripts/train_yolov11_cls.py `
  --dataset-root <lokaler_dataset_root> `
  --prepare-yolo-dataset `
  --link-method auto
```

Technischer 1-Epochen-Testlauf, nur zur Pipeline-Pruefung:

```powershell
python scripts/train_yolov11_cls.py `
  --dataset-root <lokaler_dataset_root> `
  --allow-training `
  --epochs-override 1
```

Dieser 1-Epochen-Lauf darf nicht als Modellleistung interpretiert werden. Er
prueft nur, ob Ultralytics, Datenstruktur, GPU-Zugriff, AMP und lokale
Run-Ausgaben technisch zusammenspielen.

Geplanter aussagekraeftiger YOLO-Startlauf:

```powershell
python scripts/train_yolov11_cls.py `
  --dataset-root <lokaler_dataset_root> `
  --allow-training
```

Dieser Lauf nutzt die Config `configs/experiments/yolov11_cls.yaml` mit
`yolo11n-cls`, `imgsz=320`, `batch=16`, `epochs=75`, `patience=15`,
`amp=true`, `device=0` und `workers=4`. Vor dem Start muss die lokale
YOLO-Dataset-Struktur existieren. Falls `yolo11s-cls` spaeter betrachtet wird,
darf diese Entscheidung nur nach erfolgreichem n-Modell und anhand des
Validierungssplits getroffen werden.

## 10.2 YOLO-Validation-Predictions und Metriken

Nach einem abgeschlossenen YOLOv11-cls-Lauf werden die bevorzugten Metriken
aus einer lokalen Prediction-Tabelle berechnet. Dieser Schritt startet kein
Training. Standardmaessig erlaubt das Exportskript nur den
Validierungssplit. Der Testsplit ist gesperrt und darf nur fuer die spaetere
finale Evaluation mit `--allow-test` explizit freigeschaltet werden.

Val-Predictions aus einem lokalen `best.pt` exportieren:

```powershell
python scripts/export_yolo_predictions.py `
  --model runs/global_classification/<run_name>/weights/best.pt `
  --manifest data/splits/bmw25_grouped_split_manifest.csv `
  --yolo-dataset-dir outputs/global_classification/yolov11_cls/yolo_dataset `
  --split val `
  --output-dir outputs/global_classification/<run_name>/evaluation_val `
  --imgsz 320 `
  --batch 16 `
  --device 0 `
  --model-name yolo11n-cls `
  --config-id <run_name> `
  --run-name <run_name> `
  --seed 42
```

Die erzeugte Datei `predictions_val.csv` enthaelt `image_id`,
`relative_path`, `split`, `true_label`, `predicted_label`, Run-Metadaten und
je eine `prob_<klasse>`-Spalte, sofern Ultralytics Wahrscheinlichkeiten
bereitstellt. Lokale Dateipfade werden nur relativ zur YOLO-Dataset-Struktur
geschrieben; absolute Dataset- oder Benutzerpfade gehoeren nicht in die
Prediction-CSV.

Val-Metriken aus der Prediction-Tabelle berechnen:

```powershell
python scripts/evaluate_predictions.py `
  --predictions outputs/global_classification/<run_name>/evaluation_val/predictions_val.csv `
  --output-dir outputs/global_classification/<run_name>/evaluation_val `
  --split val `
  --model-name yolo11n-cls `
  --config-id <run_name> `
  --split-version bmw25_grouped `
  --seed 42
```

Der Evaluator berechnet Accuracy, Balanced Accuracy, Macro-F1,
klassenweise Precision/Recall/F1 und eine Confusion Matrix. Diese Werte sind
Validierungsmetriken. Sie duerfen fuer Modellentscheidungen, Checkpoint-Wahl
und Hyperparameterentscheidungen genutzt werden, sind aber keine finale
Testleistung. Alle Prediction-, Metrik- und Confusion-Matrix-Dateien bleiben
lokal unter `outputs/` und werden nicht committed.

Die dokumentierte YOLO-Validierungsentscheidung zwischen `yolo11n-cls` und
`yolo11s-cls` liegt in [yolo_validation_decision.md](yolo_validation_decision.md).

## 10.3 DINOv3-Befehle

Die folgenden Befehle verwenden den lokalen Dataset-Root nur als
CLI-Parameter. Der Pfad wird nicht in versionierte Dateien geschrieben.

DINOv3-Dry-Run ohne Modellladen:

```powershell
python scripts/train_dinov3_head.py `
  --dataset-root <lokaler_dataset_root> `
  --dry-run
```

Modellladepruefung ohne Download:

```powershell
python scripts/train_dinov3_head.py `
  --dataset-root <lokaler_dataset_root> `
  --check-model
```

Falls die Gewichte nicht lokal verfuegbar sind, bricht der Modellcheck ab.
Ein Download ueber Hugging Face darf nur nach expliziter Entscheidung mit
`--allow-download` erfolgen. Tokens oder Credentials werden nicht im
Repository gespeichert.

Kleiner Smoke-Test fuer Backbone-Features und linearen Head:

```powershell
python scripts/train_dinov3_head.py `
  --dataset-root <lokaler_dataset_root> `
  --smoke-test `
  --max-smoke-samples 2
```

Der Smoke-Test nutzt nur sehr wenige Train/Val-Bilder und ist nicht als
Modellleistung interpretierbar. Ein echter Head-Trainingslauf wird erst nach
erneuter expliziter Bestaetigung mit `--allow-training` gestartet. Das Testset
bleibt in allen Vorbereitungsmodi unberuehrt.

## 10.4 DeiT-Tiny-Befehle

Die folgenden Befehle verwenden den lokalen Dataset-Root nur als
CLI-Parameter. Der Pfad wird nicht in versionierte Dateien geschrieben.

DeiT-Dry-Run:

```powershell
python scripts/train_deit_tiny.py `
  --dataset-root <lokaler_dataset_root> `
  --dry-run
```

Modellpruefung ohne pretrained weights:

```powershell
python scripts/train_deit_tiny.py `
  --dataset-root <lokaler_dataset_root> `
  --check-model
```

Kleiner Smoke-Test:

```powershell
python scripts/train_deit_tiny.py `
  --dataset-root <lokaler_dataset_root> `
  --smoke-test `
  --max-smoke-samples 2
```

Der Smoke-Test nutzt nur sehr wenige Train/Val-Bilder und ist nicht als
Modellleistung interpretierbar. Ein echter DeiT-Trainingslauf wird erst nach
erneuter expliziter Bestaetigung mit `--allow-training` gestartet. Das Testset
bleibt in allen Modi unberuehrt.

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
| YOLO-Dataset-Struktur | vorbereitet | Wird lokal aus dem Split-Manifest mit Symlinks oder Hardlinks erzeugt; nicht committen. |
| YOLO-Techniktest | offen | Optionaler 1-Epochen-Lauf nur nach explizitem Start; keine Ergebnisinterpretation. |
| DINOv3-Gewichtsquelle | offen | Lokale Bereitstellung oder erlaubter Download muss vor Training geklaert werden. |
| DINOv3-Kopfarchitektur | offen | Linearer Kopf als Startpunkt; MLP nur nach Validierungsbegruendung. |
| DeiT-Trainingsdauer | offen | From-scratch kann instabil sein; Epochen und Regularisierung nur ueber Validierung festlegen. |
| Augmentierungen | offen | Flips/Rotationen muessen fachlich plausibilisiert werden. |
| Anzahl Seeds | offen | Mehrere Seeds empfohlen, wenn Ressourcen reichen. |
| Hardwarebudget | offen | Batch Size und Laufzeit erst nach Smoke-Tests finalisieren. |
