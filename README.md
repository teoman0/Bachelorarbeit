# Vision-Transformer-Modelle zur Schleifgradbewertung metallischer Oberflächen

## Überblick

Dieses Repository begleitet eine Bachelorarbeit zur globalen Bildklassifikation
von vier Bearbeitungszuständen metallischer Oberflächen. Verglichen werden eine
YOLO-basierte Klassifikationsbaseline und mehrere Vision-Transformer-Ansätze.
Ergänzend untersucht eine rechteckbasierte lokale Analyse manuell annotierte
Bildregionen. Diese lokale Aufgabe ist keine semantische Segmentierung.

Das Repository enthält den versionierten Code, Experimentkonfigurationen,
Tests, das gruppierte Split-Manifest und die methodische Dokumentation. Rohdaten,
Modellgewichte, Checkpoints und Experimentoutputs sind nicht enthalten.

## Forschungsgegenstand

Die Aufgabe ist die bildbasierte Bewertung metallischer Oberflächen anhand von
vier Klassen:

- Erste Bearbeitungsstufe
- Zweite Bearbeitungsstufe
- Fräszustand
- Finaler Zustand

Untersucht wird insbesondere, welchen Nutzen vortrainierte Vision-Transformer-
Repräsentationen bei der begrenzten Datensatzgröße bieten. Als praxisnahe
Referenz dient YOLOv11n-cls. Eine lokale Zusatzanalyse überträgt die
Klassifikation auf manuell in CVAT markierte rechteckige Regionen.

## Untersuchte Modelle

Die globale Bildklassifikation umfasst:

- YOLOv11n-cls als vortrainierte Klassifikationsbaseline;
- DINOv3 ViT-B/16 mit eingefrorenem Backbone und linearem Head;
- DeiT-Tiny mit `pretrained=false` als ViT-from-scratch-Kontrollmodell;
- DINOv3 ViT-B/16 mit partiellem Fine-Tuning der letzten zwei
  Transformer-Blöcke, der finalen Norm und des Klassifikationskopfs.

Die lokale rechteckbasierte Analyse umfasst:

- direkte DINOv3-Inferenz auf Region-Crops;
- einen DINOv3-Region-Head für die vier Schleifgradklassen;
- einen DINOv3-Region-Head für fünf Klassen einschließlich
  `Nicht_bewertbar`.

## Zentrale Ergebnisse

Die globale finale Testauswertung verwendet 691 zuvor von der Modellentwicklung
ausgeschlossene Testbilder.
Training und Validation dienten der Parameteroptimierung, Checkpoint-Auswahl
und Modellentscheidung. Der Testsplit wurde erst nach Abschluss dieser
Entscheidungen für die finale Bewertung verwendet; aus seinen Ergebnissen
wurden keine weiteren Modellanpassungen abgeleitet.

| Modell | Accuracy | Balanced Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| YOLOv11n-cls | 0.9609 | 0.9490 | 0.9558 |
| DINOv3 frozen + Linear Head | 0.9392 | 0.9405 | 0.9397 |
| DeiT-Tiny from scratch | 0.6527 | 0.6236 | 0.6275 |
| DINOv3 Partial Fine-Tuning | 0.9493 | 0.9508 | 0.9518 |

| Lokales Modell | Testregionen | Macro-F1 |
| --- | ---: | ---: |
| DINOv3 Region-Head, vier Klassen | 48 | 0.7902 |
| DINOv3 Region-Head, fünf Klassen | 58 | 0.5722 |

YOLOv11n-cls erreicht die höchste Accuracy und den höchsten Macro-F1. DINOv3
Partial Fine-Tuning erzielt die höchste Balanced Accuracy und ist der stärkste
transformerbasierte Ansatz. DeiT-Tiny from scratch bleibt deutlich hinter den
vortrainierten Modellen zurück. Die lokale Klassifikation ist grundsätzlich
möglich, stellt wegen der wenigen rechteckigen Annotationen jedoch keine
Segmentierungsbenchmark dar.

## Datensatz und Split

Der geprüfte Datensatz umfasst 4.607 lesbare Bilder. Ein gruppierter
70/15/15-Split hält die q1- bis q4-Viertel derselben Ursprungsaufnahme stets im
gleichen Teilbestand.

| Split | Bilder | Gruppen |
| --- | ---: | ---: |
| Training | 3.225 | 1.083 |
| Validation | 691 | 232 |
| Test | 691 | 232 |
| **Gesamt** | **4.607** | **1.547** |

Alle Bilder sind genau einem Split zugeordnet; keine Gruppen-ID tritt in
mehreren Splits auf. Die Rohbilder sind nicht Bestandteil des Repositorys.
Details zu Klassen, Bildstruktur, Gruppenbildung und Datenqualität stehen in
der [Datensatzbeschreibung](docs/dataset.md).

## Repository-Struktur

```text
configs/experiments/  Versionierte Konfigurationen der ausgeführten Experimente
data/splits/          Gruppiertes Split-Manifest und kleine Split-Zusammenfassung
docs/                 Kanonische Methodik-, Ergebnis- und Reproduktionsdokumente
scripts/              Audit-, Trainings-, Inferenz- und Evaluationsskripte
src/bachelorarbeit/   Wiederverwendbare Daten- und Trainingslogik
tests/                Integritäts-, Leakage- und Sicherheitsprüfungen
```

## Installation

Die ausgeführten Experimente verwendeten Python 3.12.13. Nach dem Klonen kann
eine virtuelle Umgebung angelegt werden:

```powershell
git clone https://github.com/teoman0/Bachelorarbeit.git
cd Bachelorarbeit
python -m venv .venv
```

Zuerst müssen die in `requirements.txt` gepinnten öffentlichen Versionen
`torch 2.11.0` und `torchvision 0.26.0` über den zur Zielhardware passenden
Index der
[offiziellen PyTorch-Installationsauswahl](https://pytorch.org/get-started/locally/)
installiert werden. Der dokumentierte lokale GPU-Stand war
`torch 2.11.0+cu128` mit `torchvision 0.26.0+cu128`; daraus folgt keine
allgemeine CUDA-Vorgabe für andere Systeme. Ein passender Build mit lokalem
Suffix wie `+cu128` erfüllt die öffentlichen Versionspins und wird durch den
folgenden Installationsschritt nicht ersetzt. Danach werden die übrigen
direkten Laufzeitabhängigkeiten installiert:

```powershell
python -m pip install -r requirements.txt
```

## Daten und Checkpoints

Rohbilder, CVAT-Arbeitsdaten, Modellcheckpoints, externe Basisgewichte,
Predictions, Metrikdateien und Visualisierungen bleiben lokal. Dataset-Roots,
manuelle Annotationen und Checkpoints werden den Skripten über Configs oder
Kommandozeilenparameter übergeben; absolute lokale Pfade werden nicht
versioniert.

Ohne Originaldaten und die verwendeten Checkpoints ist eine vollständige
numerische Reproduktion der berichteten Ergebnisse nicht möglich. Der
Datenfluss, die Sicherheitsgrenzen, Konfigurationen und Auswertungsschritte
bleiben anhand des Repositorys nachvollziehbar.

## Reproduktion und Workflows

Die wichtigsten Einstiegspunkte sind:

- Dataset-Audit: `scripts/audit_dataset.py`
- gruppierte Split-Erzeugung: `scripts/create_grouped_split.py`
- globale Trainingsläufe: `scripts/train_yolov11_cls.py`,
  `scripts/train_dinov3_head.py`, `scripts/train_deit_tiny.py` und
  `scripts/train_dinov3_partial_finetune.py`
- CVAT-Aufbereitung: `scripts/prepare_cvat_region_annotations.py`
- lokale Regionenauswertung und Region-Head:
  `scripts/evaluate_dinov3_regions.py` und
  `scripts/train_dinov3_region_head.py`
- finale Testauswertung: `scripts/run_final_test_evaluation.py`

Die finale Testinferenz ist gesperrt und erfordert die ausdrückliche Option
`--allow-final-test`:

```powershell
python scripts/run_final_test_evaluation.py --config configs/experiments/final_test_evaluation.yaml --dataset-root "<dataset-root>" --manual-root "<manual-root>" --allow-final-test
```

Geprüfte Aufrufe, erforderliche lokale Eingaben und erzeugte Artefakte sind
ausschließlich in der [Reproduzierbarkeitsanleitung](docs/reproducibility.md)
ausführlich beschrieben.

## Dokumentation

- [Datensatz und gruppierter Split](docs/dataset.md)
- [Methodik](docs/methodology.md)
- [Experimenteller Aufbau](docs/experimental_setup.md)
- [Validierungsergebnisse und Modellwahl](docs/validation_results.md)
- [Rechteckbasierte Regionenanalyse](docs/region_analysis.md)
- [Reproduzierbarkeit](docs/reproducibility.md)
- [Finale Testauswertung](docs/final_test_evaluation_result.md)
- [Modelle und Drittanbieter-Lizenzen](docs/model_licenses.md)

## Grenzen

- Der verwendete Bilddatensatz ist nicht im Repository enthalten; seine
  Bereitstellung und Freigabe sind separat zu klären.
- Die verwendeten Projektcheckpoints und externen Basisgewichte sind nicht
  enthalten.
- Die lokale Analyse basiert auf einer kleinen Zahl manueller Bounding Boxes.
- Es liegen keine pixelgenauen Segmentierungsmasken vor.
- Einzelne identische Bildpaare besitzen widersprüchliche globale Labels.
- Das Repository allein ermöglicht deshalb keine vollständige numerische
  Ergebnisreproduktion.

## Lizenz und verwendete Drittkomponenten

Für den eigenen Repository-Code ist aktuell keine separate Nutzungslizenz
ausgewiesen. Drittanbieter-Code, Python-Pakete und Modellgewichte unterliegen
ihren jeweiligen Lizenzen. Die konkreten Quellen, Lizenzstände und offenen
Prüfpunkte sind in der
[Modell- und Lizenzdokumentation](docs/model_licenses.md) festgehalten. Aus der
wissenschaftlichen Nutzung in dieser Arbeit folgt keine pauschale Freigabe für
eine industrielle oder kommerzielle Weiterverwendung.
