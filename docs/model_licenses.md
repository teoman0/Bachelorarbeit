# Modell- und Lizenzdokumentation

Stand: 2026-06-16

Diese Datei dokumentiert die Lizenzlage der geplanten Modellgruppen. Sie ersetzt keine Rechtsberatung, dient aber als reproduzierbare Arbeitsnotiz fuer die Bachelorarbeit. Vor jedem Experiment muss die konkret verwendete Paketversion, Modellvariante und Gewichtsquelle ergaenzt werden.

## Kurzuebersicht

| Modellgruppe | Voraussichtliche Quelle | Lizenzstatus | Konsequenz fuer dieses Repo |
| --- | --- | --- | --- |
| YOLOv11-cls | Ultralytics `ultralytics` | AGPL-3.0 oder kommerzielle Ultralytics Enterprise License | Fuer wissenschaftliche Experimente moeglich, Lizenzhinweise dokumentieren; keine Gewichte ins Repo |
| DINOv3 | Meta / facebookresearch DINOv3 | DINOv3 License, eigene Meta-Lizenz | Nicht als lizenzsicherer Open-Source-Vergleich behandeln; konkrete Nutzungsbedingungen dokumentieren |
| DeiT-Tiny from scratch | facebookresearch/deit bzw. `timm` | Code Apache-2.0; `pretrained=False` vermeidet externe Gewichte | Geplanter lizenzbewusster ViT-Kontrollvergleich |
| ViT ueber `timm` | huggingface/pytorch-image-models | Code Apache-2.0; Gewichte koennen abweichen | Fuer jedes konkrete `timm`-Modell die Gewichtsquelle separat dokumentieren; fuer den geplanten DeiT-Tiny-Lauf keine externen Gewichte nutzen |

## YOLOv11-cls / Ultralytics

Quelle:

- https://github.com/ultralytics/ultralytics
- https://docs.ultralytics.com/

Notizen:

- Das Ultralytics-Repository ist als AGPL-3.0 lizenziert.
- Ultralytics weist zusaetzlich auf eine Enterprise License fuer kommerzielle oder produktionsnahe Nutzung hin.
- Fuer diese Bachelorarbeit ist wichtig, die verwendete Paketversion und Modellvariante zu dokumentieren.
- Lokale Runs, Checkpoints, heruntergeladene Gewichte und `runs/`-Ordner duerfen nicht in GitHub eingecheckt werden.

Konkrete Nutzung:

```text
Experiment-ID: yolo_rectangle_detection_pilot_v1
Modellgruppe: YOLO Detection / rechteckbasierte lokale Segmentierung
Modellname: yolo11n.pt
Paket/Repository: ultralytics
Paketversion oder Commit: ultralytics 8.4.90 im ersten lokalen Lauf vom 2026-07-08
Gewichtsquelle: automatischer Download durch Ultralytics beim ersten Laden von yolo11n.pt
Download-Datum: 2026-07-08
Code-Lizenz: AGPL-3.0 laut Ultralytics-Repository
Gewichte-Lizenz: Ultralytics/YOLO-Modellbedingungen laut Quelle pruefen; nicht ins Repo einchecken
Pretraining-Datensatz: COCO Detection fuer das vortrainierte Detection-Modell
Zitationshinweis: Ultralytics YOLO Dokumentation/Repository
Lokaler Gewichtepfad ausserhalb des Repos: Ultralytics-/Torch-Cache oder lokaler Gewichteordner ausserhalb von Git
Bemerkungen: Pilotlauf auf CVAT-Rechtecken aus manual_all_yolo, keine pixelgenaue Segmentierung.
```

## DINOv3

Quelle:

- https://github.com/facebookresearch/dinov3
- https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md

Notizen:

- DINOv3-Code und Modellgewichte stehen laut Repository unter der DINOv3 License.
- Die DINOv3 License ist eine eigene Meta-Lizenz und sollte nicht automatisch als unkritische Open-Source-Lizenz behandelt werden.
- Fuer Experimente muss dokumentiert werden:
  - exakter Modellname,
  - Quelle der Gewichte,
  - Zeitpunkt des Downloads,
  - akzeptierte Lizenzbedingungen,
  - lokale Speicherposition ausserhalb des Git-Repositories.
- DINOv3 ist deshalb nicht die Modellgruppe fuer den "lizenzsicheren Open-Source Vision Transformer", sondern eine eigene zu untersuchende Modellgruppe.

Konkrete Nutzung:

```text
Experiment-ID: dinov3_patch_smoke_test
Modellgruppe: DINOv3 patchbasiert
Modellname: facebook/dinov3-vits16-pretrain-lvd1689m
Paket/Repository: Hugging Face transformers
Paketversion oder Commit: transformers 5.12.1, torch 2.12.1+cpu
Gewichtsquelle: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
Download-Datum: 2026-06-18
Code-Lizenz: DINOv3 License laut Modellkarte/Repository
Gewichte-Lizenz: DINOv3 License laut Modellkarte/Repository
Pretraining-Datensatz: LVD-1689M laut Modellkarte
Zitationshinweis: DINOv3, arXiv:2508.10104
Lokaler Gewichtepfad ausserhalb des Repos: Hugging-Face-Cache ausserhalb des Repositories, z. B. `%USERPROFILE%\.cache\huggingface\hub\models--facebook--dinov3-vits16-pretrain-lvd1689m`
Bemerkungen: Nur technischer Smoke-Test ohne Training, Labels oder Bewertung.
```

```text
Experiment-ID: dinov3_bmw25_prototype_heatmaps
Modellgruppe: DINOv3 patchbasiert
Modellname: facebook/dinov3-vits16-pretrain-lvd1689m
Paket/Repository: Hugging Face transformers
Paketversion oder Commit: siehe requirements-dinov3-prototype.txt
Gewichtsquelle: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
Download-Datum: 2026-06-18
Code-Lizenz: DINOv3 License laut Modellkarte/Repository
Gewichte-Lizenz: DINOv3 License laut Modellkarte/Repository
Pretraining-Datensatz: LVD-1689M laut Modellkarte
Zitationshinweis: DINOv3, arXiv:2508.10104
Lokaler Gewichtepfad ausserhalb des Repos: Hugging-Face-Cache ausserhalb des Repositories
Bemerkungen: Qualitative Prototyp-Heatmaps ohne Training, ohne Pixelmasken und ohne finale Testset-Bewertung.
```

```text
Experiment-ID: weak_segmentation_feasibility_manual_v1
Modellgruppe: DINOv3 patchbasiert / weak segmentation feasibility
Modellname: facebook/dinov3-vits16-pretrain-lvd1689m
Paket/Repository: Hugging Face transformers
Paketversion oder Commit: siehe summary.json des lokalen Laufs
Gewichtsquelle: https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
Download-Datum: bereits lokal gecacht, Lauf am 2026-07-08 mit local_files_only=true
Code-Lizenz: DINOv3 License laut Modellkarte/Repository
Gewichte-Lizenz: DINOv3 License laut Modellkarte/Repository
Pretraining-Datensatz: LVD-1689M laut Modellkarte
Zitationshinweis: DINOv3, arXiv:2508.10104
Lokaler Gewichtepfad ausserhalb des Repos: Hugging-Face-Cache ausserhalb des Repositories
Bemerkungen: Frozen Feature Extractor, leave-one-image-out nearest-prototype,
keine Gewichtsanpassung, kein echtes Segmentierungsmodell, Rechtecke nur als schwache Masken.
```

## DeiT-Tiny from scratch

Quelle:

- https://github.com/facebookresearch/deit
- https://github.com/huggingface/pytorch-image-models

Notizen:

- Das offizielle DeiT-Repository ist unter Apache-2.0 veroeffentlicht.
- Fuer die geplante Modellmatrix ist bevorzugt `deit_tiny_patch16_224` vorgesehen.
- Der Lauf soll mit `pretrained=False` erfolgen, um externe Modellgewichte und deren separate Lizenzfragen zu vermeiden.
- DeiT-Tiny from scratch ist als Architekturkontrolle bzw. ViT-from-scratch-Untergrenze geplant, nicht als direkter Leistungsfavorit gegenueber DINOv3.
- Vor dem Experiment muessen die konkrete Implementierungsquelle, `timm`-Version und Zitationshinweise dokumentiert werden.

## ViT ueber timm

Quelle:

- https://github.com/huggingface/pytorch-image-models
- https://huggingface.co/docs/timm

Notizen:

- Der `timm`-Code ist Apache-2.0 lizenziert.
- Die Lizenz oder Nutzbarkeit einzelner vortrainierter Gewichte kann von der Gewichtsquelle und dem Pretraining-Datensatz abhaengen.
- Fuer jedes verwendete Modell muss dokumentiert werden:
  - `timm`-Modellname,
  - `timm`-Version,
  - Quelle der Gewichte,
  - Lizenz der Gewichte,
  - Pretraining-Datensatz, soweit bekannt,
  - Zitation.

## Template fuer konkrete Experimente

```text
Experiment-ID:
Modellgruppe:
Modellname:
Paket/Repository:
Paketversion oder Commit:
Gewichtsquelle:
Download-Datum:
Code-Lizenz:
Gewichte-Lizenz:
Pretraining-Datensatz:
Zitationshinweis:
Lokaler Gewichtepfad ausserhalb des Repos:
Bemerkungen:
```
