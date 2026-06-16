# Experimentplan

## Ziel

Der Experimentplan beschreibt die vorgesehene Reihenfolge der Arbeit. Er ist bewusst methodisch formuliert und enthaelt noch keinen Trainingscode.

## Phase 1: Datensatzpruefung

Vor jeder Implementierung von Trainingscode werden folgende Punkte geklaert:

- Welche Schleifgradklassen existieren?
- Wie viele Bilder gibt es pro Klasse?
- Gibt es Bauteil-, Proben-, Serien- oder Aufnahme-IDs?
- Gibt es mehrere Bilder desselben Bauteils oder derselben Aufnahme?
- Sind Bilder ueberlappend oder aus groesseren Ursprungsbildern ausgeschnitten?
- Sind Metadaten vorhanden, die fuer gruppierte Splits genutzt werden koennen?

Ergebnis dieser Phase ist eine dokumentierte Split-Strategie und, falls moeglich, ein versioniertes Split-Manifest.

## Phase 2: Split-Definition

Die Daten werden in Train, Validation und Test aufgeteilt. Die Aufteilung erfolgt vor jeder Patch-Erzeugung.

Prioritaet der Split-Ebene:

1. Bauteil-, Proben- oder Werkstueckebene, falls vorhanden.
2. Aufnahme- oder Serienebene, falls mehrere Aufnahmen zusammengehoeren.
3. Originalbildebene, falls keine hoeherwertige Gruppierung vorhanden ist.

Patches duerfen niemals dazu fuehren, dass visuell verwandte Ausschnitte desselben Originalbildes in verschiedenen Splits landen.

## Phase 3: Globale Bildklassifikation

### Experimentgruppe A: YOLOv11-cls

Ziel ist eine praxisnahe Baseline fuer globale Bildklassifikation. YOLOv11-cls wird als moderner, anwendungsnaher Vergleichspunkt betrachtet.

Zu dokumentieren:

- verwendete Ultralytics-Version,
- Modellvariante,
- Eingabegroesse,
- Augmentierungen,
- Seed,
- Trainingsdauer,
- verwendete Split-Manifeste,
- Lizenzhinweise.

### Experimentgruppe B: DINOv3 + Klassifikationskopf

DINOv3 wird zuerst eingefroren als Feature Extractor verwendet. Trainiert wird nur ein normaler Klassifikationskopf.

Zu klaeren:

- konkrete DINOv3-Backbone-Variante,
- Quelle der Modellgewichte,
- ob und welche Preprocessing-Schritte durch die Modellquelle vorgegeben sind,
- ob Features offline gecached werden duerfen und wo dieser Cache lokal abgelegt wird.

### Experimentgruppe C: Lizenzsicherer Open-Source Vision Transformer

Als lizenzsicherer Vergleich soll bevorzugt DeiT oder ein ViT ueber `timm` eingesetzt werden. Die konkrete Modell- und Gewichtsquelle wird vor dem Experiment in [model_licenses.md](model_licenses.md) dokumentiert.

Zu dokumentieren:

- Modellname in `timm` oder offizieller Modellquelle,
- Code-Lizenz,
- Gewichte-Lizenz,
- Datensatz der Vortrainierung, soweit bekannt,
- Zitationshinweise.

## Phase 4: Patchbasierte lokale Klassifikation

Die patchbasierte lokale Klassifikation erfolgt erst nach der globalen Baseline und nach festgelegtem Split.

Vorgehen:

1. Originalbilder werden zunaechst den Splits zugeordnet.
2. Patches werden innerhalb jedes Splits separat erzeugt.
3. Patchlabels werden aus dem Bildlabel abgeleitet.
4. Ein Patchklassifikator wird trainiert oder ein globales Modell wird patchweise angewendet.
5. Patchvorhersagen werden an ihre Bildposition zurueckprojiziert.
6. Visualisierungen werden als Heatmaps interpretiert, nicht als Ground-Truth-Segmentierung.

Wichtige Einschraenkung: Patchlabels koennen verrauscht sein, weil sie nicht pixelgenau annotiert wurden. Ergebnisse muessen deshalb als lokale Klassifikationshinweise und nicht als echte semantische Segmentierung formuliert werden.

## Metriken

Fuer die globale Klassifikation sind mindestens vorgesehen:

- Accuracy,
- Balanced Accuracy,
- Macro F1,
- Confusion Matrix,
- pro-Klasse Precision und Recall.

Falls die Klassen ordinal interpretierbar sind, koennen zusaetzlich Fehlerabstaende zwischen Schleifgraden betrachtet werden. Diese Entscheidung muss vorher dokumentiert werden.

## Config-First-Prinzip

Jedes Experiment soll ueber eine Config-Datei in `configs/` reproduzierbar sein. Eine Config soll spaeter mindestens enthalten:

- Experimentname,
- Datenpfade oder Dataset-ID,
- Split-Manifest,
- Modellgruppe und Modellvariante,
- Bildgroesse,
- Augmentierungen,
- Optimizer- und Scheduler-Parameter,
- Seed,
- Output-Verzeichnis,
- Lizenz- und Quellenhinweise fuer Modellgewichte.

## Output-Regeln

Lokale Outputs gehoeren nach `outputs/` und werden nicht eingecheckt. Kuratierte Tabellen und Abbildungen, die fuer die Thesis benoetigt werden und keine grossen Dateien sind, koennen nach `reports/tables/` bzw. `reports/figures/` uebernommen werden.

Checkpoints und heruntergeladene Modellgewichte bleiben ausserhalb von Git.
