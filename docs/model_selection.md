# Modellauswahl

Stand: 2026-07-14

Diese Datei dokumentiert die geplante Modellauswahl fuer Kapitel 3.2
"Modellauswahl". Sie legt die methodische Rolle der Modellgruppen fest, schreibt
aber noch keine konkrete Trainingsimplementierung vor. Alle spaeteren
Experimente muessen weiterhin ueber Config-Dateien in `configs/` gesteuert und
mit Split-Manifest, Seed, Git-Commit, Paketversionen und Lizenznotizen
reproduzierbar dokumentiert werden.

## Auswahlkriterien

Die Modellauswahl orientiert sich an folgenden Kriterien:

- Bezug zur Forschungsfrage: Die Modelle muessen geeignet sein, den Schleifgrad
  metallischer Oberflaechen aus Bilddaten vorherzusagen.
- Eignung fuer Bildklassifikation: Der Hauptvergleich betrachtet globale
  Bildklassifikation, also die Zuordnung eines gesamten Bildes zu einer von vier
  Schleifgradklassen.
- Umgang mit kleinen bzw. industriellen Datensaetzen: Die Auswahl soll sowohl
  praxisnahe Baselines als auch vortrainierte Repraesentationen und eine
  ViT-Kontrolle ohne externe Gewichte abdecken.
- Reproduzierbarkeit: Modellvariante, Preprocessing, Split-Manifest,
  Hyperparameter, Seed, Paketversionen und Outputs muessen nachvollziehbar
  dokumentiert werden.
- Lizenz- und Nutzungsrisiken: Modelle mit eigener oder restriktiver Lizenz
  werden nicht als unkritische Open-Source-Vergleiche dargestellt.
- Hardwareaufwand: Bevorzugt werden kleine Varianten, die auf der verfuegbaren
  lokalen Hardware realistisch trainierbar oder auswertbar sind.

## Geplante Modellmatrix

| Rolle | Modell | Trainingssetup | Zweck | Lizenznotiz | Hauptauswertungsebene |
| --- | --- | --- | --- | --- | --- |
| Praxisnahe moderne Baseline | YOLOv11-cls | Supervised Klassifikation; exakte Variante spaeter, bevorzugt `yolo11n-cls` oder `yolo11s-cls` je nach Hardware | Starker anwendungsnaher Vergleichspunkt fuer Bildklassifikation | Ultralytics AGPL-3.0 / Enterprise-Kontext dokumentieren; nicht als unkritischen Open-Source-Vergleich formulieren | Globales Bild |
| Hauptansatz | DINOv3 frozen + Klassifikationskopf | DINOv3-Backbone zunaechst eingefroren; trainiert wird nur ein linearer oder kleiner Klassifikationskopf | Pruefen, ob self-supervised vortrainierte Repraesentationen den Schleifgrad robust abbilden | Eigene DINOv3 License dokumentieren; nicht als unkritischen Open-Source-Vergleich behandeln | Globales Bild |
| Architekturkontrolle / lizenzbewusster ViT | DeiT-Tiny from scratch | Bevorzugt `deit_tiny_patch16_224` mit `pretrained=False`; Training ohne externe Gewichte | Abschaetzen, was ein kleiner ViT ohne grosses Vortraining auf dem Datensatz leisten kann | DeiT bzw. `timm` als Apache-2.0-Codebasis dokumentieren; durch `pretrained=False` werden externe Gewichts-Lizenzfragen vermieden | Globales Bild |
| Zusatzanalyse | DINOv3 patchbasiert | Split zuerst auf Gruppenebene; danach Patches; DINOv3 pro Patch; Rueckprojektion als Heatmap | Qualitative lokale Schleifgradkarten und raeumliche Plausibilisierung | DINOv3 License dokumentieren; Patch-Outputs nicht als Segmentierungs-Ground-Truth darstellen | Patch / Heatmap, nicht Hauptmodellvergleich |

## Hauptvergleich

Der Hauptvergleich der Arbeit erfolgt auf globaler Bildebene. Dabei erhalten
YOLOv11-cls, DINOv3 frozen + Klassifikationskopf und DeiT-Tiny from scratch
jeweils ein gesamtes Bild als Eingabe und geben eine von vier
Schleifgradklassen aus.

YOLOv11-cls dient als praxisnahe moderne Baseline. Es wird in dieser Arbeit als
Klassifikationsmodell genutzt, nicht als Detektions-, Segmentierungs- oder
Lokalisierungsmodell. Die konkrete YOLOv11-Variante wird erst vor dem Training
auf Basis des Hardwarelimits festgelegt; kleine Varianten wie `yolo11n-cls` oder
`yolo11s-cls` sind bevorzugt.

DINOv3 frozen + Klassifikationskopf ist der zentrale methodische Ansatz. Der
DINOv3-Backbone wird zunaechst nur als eingefrorener Feature Extractor
verwendet. Trainiert wird ein einfacher Klassifikationskopf. Ein spaeteres
Fine-Tuning des Backbones ist nur optional und nur dann vorgesehen, wenn die
Validierungsergebnisse eine saubere Begruendung liefern. Das Testset darf dafuer
nicht verwendet werden.

DeiT-Tiny from scratch dient als Architekturkontrolle. Das Modell soll zeigen,
welche Leistung ein kleiner Vision Transformer ohne grosse externe
Vortrainierung auf dem verfuegbaren Datensatz erreichen kann. Es wird deshalb
nicht als direkter Leistungsfavorit gegenueber DINOv3 formuliert, sondern als
ViT-from-scratch-Untergrenze bzw. Kontrollmodell.

## Patchbasierte Zusatzanalyse

DINOv3 patchbasiert ist eine Zusatzanalyse zur raeumlichen Visualisierung und
kein Teil des primaeren globalen Modellvergleichs. Dabei werden Bilder erst nach
einem gruppierten Train/Val/Test-Split in Patches zerlegt. Patches aus demselben
Originalbild, Bauteil oder derselben Aufnahmeserie duerfen nicht ueber
verschiedene Splits verteilt werden.

Die patchbasierte Auswertung ist keine echte semantische Segmentierung, weil
keine pixelgenauen Masken vorliegen. Patchlabels werden aus globalen Bildlabels
abgeleitet und koennen deshalb lokal verrauscht sein, insbesondere wenn ein Bild
heterogene Oberflaechenbereiche enthaelt. Die resultierenden Heatmaps werden
daher als qualitative lokale Klassifikationskarten interpretiert, nicht als
Ground-Truth-Segmentierungen.

## Offene Entscheidungen

Vor dem Training muessen folgende Punkte final festgelegt und dokumentiert
werden:

- exakte YOLOv11-Variante, zum Beispiel `yolo11n-cls` oder `yolo11s-cls`;
- DINOv3-Backbone-Variante;
- Bildgroesse und Preprocessing je Modellgruppe;
- Patchgroesse und Stride fuer die patchbasierte Zusatzanalyse;
- Hardwarelimit, insbesondere verfuegbarer GPU-/CPU-Speicher und Laufzeitbudget;
- ob DeiT-Tiny from scratch auf dem Datensatz ausreichend stabil trainierbar ist.
