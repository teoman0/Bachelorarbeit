# Modell- und Lizenzdokumentation

Stand und Abrufdatum der verlinkten Upstream-Quellen: 2026-07-21.

Diese Übersicht dokumentiert die tatsächlich verwendeten Modell- und
Softwarekomponenten. Sie ist keine Rechtsberatung. Vor einer Veröffentlichung
von Gewichten, einer Weitergabe des Codes oder einem praktischen Einsatz sind
die dann gültigen Originalbedingungen erneut zu prüfen. Für den eigenen Code
dieses Repositorys ist derzeit keine separate Nutzungslizenz ausgewiesen.

## Modelle und zentrale Frameworks

| Komponente | Konkrete Verwendung | Version | Code-Lizenz | Modellgewichte | Offizielle Quellen |
| --- | --- | --- | --- | --- | --- |
| Ultralytics YOLO | `yolo11n-cls` als vortrainierte globale Baseline | `ultralytics 8.4.90` | AGPL-3.0; Ultralytics bietet alternativ eigene Enterprise-Bedingungen an | Ultralytics ordnet auch seine trainierten Modelle standardmäßig der AGPL-3.0 zu; eine abweichende Enterprise-Vereinbarung wurde für dieses Repository nicht dokumentiert | [Ultralytics-Lizenzseite](https://www.ultralytics.com/license), [Upstream-Lizenz](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` für Frozen Head, Partial Fine-Tuning und Region-Heads | Modellstand aus dem lokalen Hugging-Face-Cache; geladen mit `transformers 5.12.1` | Eigene DINOv3 License für die von Meta bereitgestellten DINO-Materialien | Die DINOv3 License umfasst ausdrücklich auch trainierte Gewichte; der Modellzugang ist auf Hugging Face an eine Zugangsvereinbarung gebunden | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), [offizielle Modellkarte](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), [Repository](https://github.com/facebookresearch/dinov3) |
| Hugging Face Transformers | Laden und Ausführen des DINOv3-Backbones | `5.12.1` | Apache-2.0 | Keine eigenen Gewichte; die geladene DINOv3-Datei behält ihre separate Lizenz | [Upstream-Lizenz](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| timm | Implementierung von `deit_tiny_patch16_224` | `1.0.28` | Apache-2.0 | Im Experiment wurde `pretrained=false` verwendet; es wurden keine timm- oder DeiT-Basisgewichte geladen | [timm-Lizenz](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE), [timm-Repository](https://github.com/huggingface/pytorch-image-models) |
| DeiT | Architektur- und Publikationsbezug des from-scratch-Kontrollmodells; die konkrete Implementierung stammt aus timm | keine separate Laufzeitkomponente | Das offizielle DeiT-Repository steht unter Apache-2.0 | Keine externen DeiT-Gewichte verwendet | [offizielle DeiT-Lizenz](https://github.com/facebookresearch/deit/blob/main/LICENSE), [DeiT-Repository](https://github.com/facebookresearch/deit) |
| PyTorch | Trainings- und Inferenzlaufzeit | `2.11.0+cu128` im dokumentierten GPU-Stand | BSD-3-Clause | Keine vom PyTorch-Projekt bereitgestellten Modellgewichte verwendet | [PyTorch-Lizenz](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | Laufzeitabhängigkeit des Bildmodell-Stacks | `0.26.0+cu128` im dokumentierten GPU-Stand | BSD-3-Clause | Keine torchvision-Gewichte verwendet | [torchvision-Lizenz](https://github.com/pytorch/vision/blob/main/LICENSE) |

Die externen YOLO- und DINOv3-Basisgewichte sowie alle daraus erzeugten
Projektcheckpoints sind nicht Bestandteil des Repositorys. Bei Ultralytics
sind insbesondere die AGPL-Pflichten und die angebotene Enterprise-Alternative
vor einer nicht vollständig offenen Weiterverwendung zu prüfen. Bei DINOv3
gelten eigene Vertragsbedingungen für Code und Gewichte; Weitergabe,
Publikationshinweise und praktische Nutzung müssen gegen die jeweils aktuelle
DINOv3 License geprüft werden.

## Weitere direkte Laufzeitabhängigkeiten

| Paket | Verwendung | Verwendete Version | Lizenz | Offizielle Quelle |
| --- | --- | ---: | --- | --- |
| NumPy | numerische Verarbeitung und Prediction-Tabellen | `2.4.6` | BSD-3-Clause mit gebündelten Drittanbieterhinweisen | [NumPy-Lizenz](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pandas | CSV-basierte Evaluation | `3.0.3` | BSD-3-Clause | [pandas-Lizenz](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| Pillow | Bilddekodierung, EXIF-Transpose, Crops und Visualisierungen | `12.2.0` | MIT-CMU | [Pillow-Lizenz](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| Matplotlib | Abbildungen des Datensatzaudits | `3.11.0` | Matplotlib License | [Matplotlib-Lizenz](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| PyYAML | Laden versionierter Experimentkonfigurationen | `6.0.3` | MIT | [PyYAML-Lizenz](https://github.com/yaml/pyyaml/blob/main/LICENSE) |

Die angegebenen Versionen stammen aus der für die finalen Workflows
verwendeten lokalen Python-3.12.13-Umgebung und den gespeicherten
Run-Metadaten. Abhängigkeiten, die diese Pakete transitiv installieren,
unterliegen zusätzlich ihren eigenen Lizenztexten.

Diese Übersicht führt die vom eigenen finalen Code direkt verwendeten Pakete
und die für die dokumentierten Modell-CLIs erforderlichen zentralen
Frameworks. Transitive Abhängigkeiten werden nicht einzeln aufgelistet; ihre
Lizenztexte bleiben bei einer Weitergabe der installierten Laufzeitumgebung
zusätzlich zu beachten.

## Verbleibende Unsicherheiten

Die exakten Upstream-Revisionen, Datei-Hashes und ursprünglichen
Downloadzeitpunkte der lokal verwendeten YOLO- und DINOv3-Basisgewichte sind
nicht im versionierten Repository festgehalten. Da auch die Gewichtsdateien
nicht enthalten sind, können diese Angaben aus einem frischen Clone nicht
nachträglich verifiziert werden. Vor einer erneuten Beschaffung oder
Weitergabe sind deshalb Modellkarte, Lizenzfassung und Artefakt-Metadaten am
konkreten Download erneut zu dokumentieren.

Ebenso ist nicht geklärt, unter welchen Bedingungen der eigene Repository-Code
weitergegeben werden darf. Die aufgeführten Drittanbieter-Lizenzen erteilen
keine Lizenz an den selbst erstellten Projektbestand oder an den verwendeten
Bilddatensatz.

## Relevanz für die Bachelorarbeit

Für die wissenschaftliche Arbeit sind insbesondere folgende Angaben
erforderlich:

- exakter Modellname und Implementierungsquelle;
- Pretraining-Status und Herkunft externer Basisgewichte;
- verwendete Paketversionen und Abrufstand;
- korrekte Zitation der Modellpublikationen und Upstream-Projekte;
- klare Kennzeichnung, dass weder Basisgewichte noch Projektcheckpoints im
  Repository veröffentlicht werden.

Diese Angaben sichern methodische Nachvollziehbarkeit und ordnen den Einsatz
fremder Komponenten ein. Sie ersetzen nicht die Prüfung konkreter
Weitergaberechte.

## Relevanz für eine praktische Weiterverwendung

Vor einer späteren Veröffentlichung, Bereitstellung als Dienst, Integration in
ein Produkt oder Weitergabe von Gewichten sind zusätzlich mindestens folgende
Punkte rechtlich zu prüfen:

- ob die konkrete Nutzung die AGPL-3.0-Bedingungen von Ultralytics erfüllt oder
  eine gesonderte Vereinbarung erfordert;
- welche Fassung der DINOv3 License und welche Zugangsbedingungen für Code,
  Basisgewichte und abgeleitete Checkpoints gelten;
- ob Lizenztexte, Copyright-Hinweise, NOTICE-Dateien oder Quellcode bei einer
  Distribution mitgeliefert werden müssen;
- ob Rechte am Bilddatensatz, an Annotationen und an erzeugten Checkpoints eine
  Veröffentlichung erlauben;
- ob die noch ungeklärte Lizenzierung des eigenen Repository-Codes eine
  Weitergabe überhaupt gestattet.

Eine pauschale industrielle oder kommerzielle Nutzbarkeit wird daraus nicht
abgeleitet. Bei Unklarheit ist vor dem Einsatz eine rechtliche Prüfung
erforderlich.
