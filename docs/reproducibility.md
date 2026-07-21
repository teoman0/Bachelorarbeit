# Reproduzierbarkeit

## Zweck und Geltungsbereich

Dieses Dokument ist die kanonische technische Anleitung zur
Nachvollziehbarkeit der versionierten Workflows. Es beschreibt Voraussetzungen,
Sicherheitsgrenzen, geprüfte CLI-Aufrufe und lokale Artefaktpfade, ohne eine
vollständige Ergebnisreproduktion ohne die nicht veröffentlichten Daten und
Checkpoints zu versprechen.

Dieses Dokument beschreibt die versionierten Workflows für Datensatzaudit,
gruppierten Split, globale Klassifikation, rechteckbasierte Regionenanalyse und
finale Evaluation. Die Befehle setzen lokal verfügbare Rohbilder und bereits
erzeugte Modellcheckpoints voraus. Weder Rohdaten noch Checkpoints,
Modellgewichte, Predictions oder Metrikartefakte sind Bestandteil des
Repositorys.

Alle unten verwendeten Kommandozeilenoptionen wurden gegen die jeweilige
`--help`-Ausgabe der aktuellen Skripte geprüft. Platzhalter in spitzen
Klammern müssen durch lokale Pfade ersetzt werden; diese Pfade dürfen nicht in
versionierte Configs oder Ergebnisdateien übernommen werden.

## Reproduzierbarkeitsebenen

Die Nachvollziehbarkeit ist in drei Ebenen zu unterscheiden:

| Ebene | Umfang |
| --- | --- |
| Code-Reproduzierbarkeit | Versionierte Skripte, Configs und Tests können in einer passenden Python-Umgebung geprüft werden. |
| Workflow-Nachvollziehbarkeit | Datenfluss, Splitregeln, Parameter, Sicherheitsflags und lokale Artefakte sind dokumentiert. |
| Vollständige Ergebnisreproduktion | Erfordert zusätzlich die nicht veröffentlichten BMW-25-Rohbilder, CVAT-Dateien, externen Basisgewichte und lokal erzeugten Projektcheckpoints. |

Ein frischer Clone ohne diese Daten und Checkpoints kann die Tests und
Dry-Runs der dateiunabhängigen Logik ausführen, aber weder die berichteten
Predictions noch die Ergebniskennzahlen vollständig reproduzieren.

## Umgebung

Die ausführbaren Experimente protokollierten Python 3.12.13. Für eine
möglichst nahe Reproduktion wird deshalb Python 3.12 empfohlen. Eine lokale
Windows-Umgebung kann wie folgt angelegt werden:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dabei muss `python` auf den installierten Python-3.12-Interpreter verweisen.

`requirements.txt` beschreibt die benötigten Bibliotheksfamilien, ist aber
kein vollständig gepinnter Lockfile. Exakte Paketstände eines Laufs sind den
lokalen `run_metadata.json`- beziehungsweise Framework-Metadaten zu entnehmen.
Für GPU-Läufe muss eine zur lokalen NVIDIA-Umgebung passende CUDA-fähige
PyTorch-Distribution nach der offiziellen PyTorch-Installationsauswahl
installiert werden. Eine konkrete CUDA-Wheel-Version wird hier nicht
festgeschrieben, da sie von Treiber und Zielsystem abhängt.

DINOv3-Skripte blockieren Downloads standardmäßig. Auf einem neuen System
kann der vortrainierte Backbone nach Prüfung von Lizenz und Netzwerkzugriff
explizit mit `--allow-download` bezogen werden. Projektspezifische
Klassifikations- und Region-Checkpoints müssen separat lokal bereitgestellt
werden.

### Import- und Laufzeitabhängigkeiten

Die dokumentierten Skripte laden `torch`, `timm`, `transformers` und
`ultralytics` erst in den Modellmodi. Ihre `--help`-Aufrufe benötigen daher
keinen bereits importierbaren Deep-Learning-Stack. Für den Modulimport werden
je nach Skript jedoch grundlegende Pakete wie Pillow, NumPy, Matplotlib oder
PyYAML benötigt.

| Workflow | Deep-Learning-Pakete ab Modellmodus |
| --- | --- |
| YOLO-Training und -Inferenz | PyTorch und Ultralytics |
| DINOv3 Check, Smoke-Test, Training und Inferenz | PyTorch und Transformers |
| DeiT Check, Smoke-Test und Training | PyTorch und timm |
| Region-Heads | PyTorch und Transformers |
| finale Gesamtauswertung | PyTorch, Ultralytics, timm und Transformers entsprechend den ausgewählten Modellen |

Datensatzaudit, Split-Erzeugung und CVAT-Tabellenvorbereitung laden keine
Deep-Learning-Modelle. Die Option `--help` verarbeitet weder Daten noch
Checkpoints.

## Erwartete lokale Datenstruktur

Der globale Dataset-Root enthält einen Unterordner je technischem Klassenlabel
und darunter die Bilddateien. Das versionierte Split-Manifest referenziert die
Bilder ausschließlich über relative Pfade.

```text
<dataset-root>/
  Erste Bearbeitungssufe Viertel/
  Finaler Zustand Viertel/
  Fräszustand Viertel/
  Zweite Bearbeitungsstufe Viertel/
```

Der lokale CVAT-Arbeitsordner enthält die Bilddateien und die vier
Metadatendateien des manuellen Exports:

```text
<manual-root>/
  images/
  annotations/cvat_annotations.json
  annotations/frame_meta.json
  annotations/labels.json
  manifest.csv
```

Die Variablen `BMW25_MANUAL_ALL_ROOT` oder die Option `--manual-root` können
den CVAT-Root bereitstellen. Absolute Pfade bleiben lokal.

## Datensatzaudit und Split

Der Audit liest Bilder und schreibt ausschließlich lokale Berichtstabellen
und Abbildungen:

```powershell
.\.venv\Scripts\python.exe scripts/audit_dataset.py --data-root "<dataset-root>" --output-dir outputs/dataset_audit
```

Der gruppierte Split wird mit Seed 42 und dem q1-bis-q4-Gruppenregex erzeugt.
Das Skript kopiert keine Bilder und schreibt nur relative Pfade in das
Manifest:

```powershell
.\.venv\Scripts\python.exe scripts/create_grouped_split.py --data-root "<dataset-root>" --output-manifest data/splits/bmw25_grouped_split_manifest.csv --summary-json data/splits/bmw25_grouped_split_summary.json --summary-md docs/dataset_split_summary.md --ratios 0.70,0.15,0.15 --seed 42
```

Vor einer Veröffentlichung des originalen Manifests ist dessen Freigabe
separat zu prüfen. Für die hier dokumentierten Experimente bleibt die
vorhandene versionierte Fassung die verbindliche Split-Referenz.

## Globale Trainingsläufe

Die Trainingsskripte führen ohne `--allow-training` keinen langen Lauf aus.
Ein `--dry-run` prüft Config, Manifest und lokale Dateien, ohne ein Modell zu
trainieren.

### YOLOv11n-cls

Zuerst wird eine lokale YOLO-Ordnerstruktur aus Links auf Train- und
Validierungsbilder erzeugt. Testlinks werden dabei nicht angelegt.

```powershell
.\.venv\Scripts\python.exe scripts/train_yolov11_cls.py --config configs/experiments/yolov11_cls.yaml --dataset-root "<dataset-root>" --prepare-yolo-dataset --link-method auto
.\.venv\Scripts\python.exe scripts/train_yolov11_cls.py --config configs/experiments/yolov11_cls.yaml --dataset-root "<dataset-root>" --allow-training
```

### DINOv3 frozen + Linear Head

```powershell
.\.venv\Scripts\python.exe scripts/train_dinov3_head.py --config configs/experiments/dinov3_linear_head.yaml --dataset-root "<dataset-root>" --dry-run
.\.venv\Scripts\python.exe scripts/train_dinov3_head.py --config configs/experiments/dinov3_linear_head.yaml --dataset-root "<dataset-root>" --allow-training
```

### DeiT-Tiny from scratch

```powershell
.\.venv\Scripts\python.exe scripts/train_deit_tiny.py --config configs/experiments/deit_tiny_scratch.yaml --dataset-root "<dataset-root>" --dry-run
.\.venv\Scripts\python.exe scripts/train_deit_tiny.py --config configs/experiments/deit_tiny_scratch.yaml --dataset-root "<dataset-root>" --allow-training
```

### DINOv3 Partial Fine-Tuning

```powershell
.\.venv\Scripts\python.exe scripts/train_dinov3_partial_finetune.py --config configs/experiments/dinov3_partial_finetune_last2.yaml --dataset-root "<dataset-root>" --dry-run
.\.venv\Scripts\python.exe scripts/train_dinov3_partial_finetune.py --config configs/experiments/dinov3_partial_finetune_last2.yaml --dataset-root "<dataset-root>" --allow-training
```

## Validierung

Die DINOv3-, DeiT- und Region-Head-Trainingsskripte exportieren die
Validierungsprädiktionen und Metriken des besten Checkpoints direkt in ihr
lokales Output-Verzeichnis. Für YOLO wird die Inferenz separat exportiert und
danach mit dem allgemeinen Evaluator ausgewertet:

```powershell
.\.venv\Scripts\python.exe scripts/export_yolo_predictions.py --model runs/global_classification/yolov11_cls_n_imgsz320_seed42/weights/best.pt --manifest data/splits/bmw25_grouped_split_manifest.csv --yolo-dataset-dir outputs/global_classification/yolov11_cls/yolo_dataset --split val --output-dir outputs/global_classification/yolov11_cls_n_imgsz320_seed42/evaluation_val --imgsz 320 --batch 16 --device 0 --run-name yolov11_cls_n_imgsz320_seed42 --seed 42 --overwrite
.\.venv\Scripts\python.exe scripts/evaluate_predictions.py --predictions outputs/global_classification/yolov11_cls_n_imgsz320_seed42/evaluation_val/predictions_val.csv --output-dir outputs/global_classification/yolov11_cls_n_imgsz320_seed42/evaluation_val --split val --class-names "Erste Bearbeitungssufe Viertel,Finaler Zustand Viertel,Fräszustand Viertel,Zweite Bearbeitungsstufe Viertel" --model-name YOLOv11n-cls --config-id yolov11_cls_n_imgsz320_seed42 --seed 42
```

## CVAT-Regionentabelle und lokale Modelle

Die vollständige Regionentabelle kann alle gematchten Splits inventarisieren.
Nachfolgende Entwicklungsworkflows filtern intern auf Train und Validation;
Testregionen werden von diesen Skripten nicht akzeptiert.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_cvat_region_annotations.py --config configs/experiments/cvat_region_analysis.yaml --manual-root "<manual-root>" --split all --include-nicht-bewertbar --exclude-unmatched --allow-export
```

Die aktuelle Export-Config schreibt nach
`outputs/cvat_region_analysis/manual_all/region_annotations.csv`. Diese
vollständige Tabelle enthält die 281 gematchten Train-, Validation- und
Testregionen. Die abgeschlossenen Entwicklungsruns hatten zuvor eine lokale
Train-/Validation-Teiltabelle mit 223 Zeilen unter
`outputs/region_analysis/cvat_region_analysis/region_annotations.csv`
geladen. Ein zeilenweiser Vergleich bestätigte, dass ihre 223 Train- und
Validation-Datensätze mit dem entsprechenden Ausschnitt der vollständigen
Tabelle übereinstimmen. Die final verwendeten Region-Configs referenzieren
deshalb einheitlich den kanonischen vollständigen Exportpfad; ihre
Entwicklungsmodi filtern weiterhin strikt auf Train und Validation.

### Direkte Regioneninferenz und Ablation

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_dinov3_regions.py --config configs/experiments/dinov3_region_eval.yaml --manual-root "<manual-root>" --region-table outputs/cvat_region_analysis/manual_all/region_annotations.csv --split val --dry-run
.\.venv\Scripts\python.exe scripts/evaluate_dinov3_regions.py --config configs/experiments/dinov3_region_eval.yaml --manual-root "<manual-root>" --region-table outputs/cvat_region_analysis/manual_all/region_annotations.csv --split val --run-ablation --include-nicht-bewertbar --allow-evaluate
```

### Vier-Klassen-Region-Head

```powershell
.\.venv\Scripts\python.exe scripts/train_dinov3_region_head.py --config configs/experiments/dinov3_region_head.yaml --manual-root "<manual-root>" --region-table outputs/cvat_region_analysis/manual_all/region_annotations.csv --dry-run
.\.venv\Scripts\python.exe scripts/train_dinov3_region_head.py --config configs/experiments/dinov3_region_head.yaml --manual-root "<manual-root>" --region-table outputs/cvat_region_analysis/manual_all/region_annotations.csv --allow-training --export-region-images --export-overlays --max-visualization-images 12
```

### Fünf-Klassen-Region-Head

```powershell
.\.venv\Scripts\python.exe scripts/train_dinov3_region_head.py --config configs/experiments/dinov3_region_head_5class.yaml --manual-root "<manual-root>" --region-table outputs/cvat_region_analysis/manual_all/region_annotations.csv --include-nicht-bewertbar --dry-run
.\.venv\Scripts\python.exe scripts/train_dinov3_region_head.py --config configs/experiments/dinov3_region_head_5class.yaml --manual-root "<manual-root>" --region-table outputs/cvat_region_analysis/manual_all/region_annotations.csv --include-nicht-bewertbar --allow-training --export-region-images --export-overlays --max-visualization-images 12
```

## Finale Testauswertung

Die finale Evaluation darf erst nach abgeschlossener Modell- und
Hyperparameterwahl ausgeführt werden. Der zentrale Workflow trainiert nicht,
erzeugt keine Splits und blockiert Testinferenz ohne das verpflichtende Flag
`--allow-final-test`.

```powershell
.\.venv\Scripts\python.exe scripts/run_final_test_evaluation.py --config configs/experiments/final_test_evaluation.yaml --dataset-root "<dataset-root>" --manual-root "<manual-root>" --allow-final-test
```

Die Config fixiert die ausgewählten Modelle und schreibt lokale Ergebnisse
unter `outputs/final_test_evaluation_reproducible/`. Nach der finalen
Auswertung dürfen Testwerte nicht für weitere Modell-, Hyperparameter- oder
Schwellenwertentscheidungen verwendet werden.

Für die vollständige finale Auswertung werden lokal folgende Checkpoints
erwartet:

| Modell | Erwarteter lokaler Checkpoint |
| --- | --- |
| YOLOv11n-cls | `runs/global_classification/yolov11_cls_n_imgsz320_seed42/weights/best.pt` |
| DINOv3 frozen + Linear Head | `outputs/global_classification/dinov3_linear_head_bmw25_seed42/checkpoints/best_head.pt` |
| DeiT-Tiny from scratch | `outputs/global_classification/deit_tiny_scratch_bmw25_seed42/checkpoints/best_model.pt` |
| DINOv3 Partial Fine-Tuning | `outputs/global_classification/dinov3_partial_finetune_last2_bmw25_seed42/checkpoints/best_model.pt` |
| DINOv3 Region-Head, vier Klassen | `outputs/region_analysis/dinov3_region_head_bmw25_seed42/checkpoints/best_model.pt` |
| DINOv3 Region-Head, fünf Klassen | `outputs/region_analysis/dinov3_region_head_5class_bmw25_seed42/checkpoints/best_model.pt` |

Keiner dieser Checkpoints und keine Rohbilddatei ist im Repository enthalten.

## Lokale Artefakte

| Workflow | Lokaler, nicht versionierter Bereich |
| --- | --- |
| Datensatzaudit | `outputs/dataset_audit/` |
| YOLO-Dataset und Validation | `outputs/global_classification/yolov11_cls/`, `runs/global_classification/` |
| globale DINOv3- und DeiT-Läufe | `outputs/global_classification/` |
| CVAT-Regionentabelle | `outputs/cvat_region_analysis/` oder explizit übergebener lokaler Pfad |
| Regioneninferenz und Region-Heads | `outputs/region_analysis/` |
| finale Evaluation | `outputs/final_test_evaluation_reproducible/` |

Jeder Lauf soll Config-ID, Seed, Git-Commit, Paketversionen, Klassenmapping und
Split-Referenz in seinen Metadaten festhalten. Die erzeugten CSV-, JSON-,
Bild- und Checkpoint-Dateien bleiben von Git ausgeschlossen.

## Offline-Prüfungen

Die Python-Quellen und Unit-Tests können ohne Training und ohne Modellinferenz
geprüft werden:

```powershell
.\.venv\Scripts\python.exe -m compileall -q scripts src
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Diese Prüfungen ersetzen keine Daten- oder Modellreproduktion, erkennen aber
Syntaxfehler und testen die versionierte Kernlogik ohne Zugriff auf den
Testsplit.

## Verwandte Dokumente

Datensatz und Splitstrategie stehen in der
[Datensatzbeschreibung](dataset.md). Die ausgeführten Modellkonfigurationen
beschreibt der [experimentelle Aufbau](experimental_setup.md); Methodik und
finale Bewertung sind in der [Methodik](methodology.md) und der [finalen
Testauswertung](final_test_evaluation_result.md) dokumentiert.
Lizenzbedingungen stehen in der [Lizenzdokumentation](model_licenses.md).
