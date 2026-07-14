# Untersuchung von Vision-Transformer-Modellen zur bildbasierten Bewertung des Schleifgrades metallischer Oberflaechen

Dieses Repository begleitet die experimentelle Durchfuehrung einer Bachelorarbeit zur bildbasierten Klassifikation des Schleifgrades metallischer Oberflaechen. Ziel ist ein reproduzierbarer Vergleich moderner Bildklassifikationsverfahren, insbesondere Vision-Transformer-basierter Modelle, unter besonderer Beachtung sauberer Datensplits und der Vermeidung von Data Leakage.

## Aktueller Status

Dieses Repository enthaelt methodische Dokumentation, Datensatz-Audit-Ergebnisse,
Modell- und Lizenznotizen, ein Trainings- und Vergleichsdesign,
Config-Templates sowie kleine Smoke-Test-Skripte. Es enthaelt keine Rohdaten,
keine finalen Trainingslaeufe, keine finalen Checkpoints und keine
Modellgewichte.

## Ziel der Arbeit

Untersucht werden Modelle, die aus Bildern metallischer Oberflaechen den Schleifgrad vorhersagen. Die Experimente sollen zeigen, welche Modellgruppe fuer die globale Bildklassifikation geeignet ist und ob patchbasierte lokale Vorhersagen eine sinnvolle Annahme fuer raeumlich aufgeloeste Schleifgradkarten liefern koennen.

## Geplante Modellgruppen

1. **YOLOv11-cls**  
   Praxisnahe Baseline fuer globale Bildklassifikation. Die Aufgabe ist
   Bild -> eine von vier Schleifgradklassen. YOLOv11-cls wird hier nicht als
   Segmentierungsmodell genutzt.

2. **DINOv3 frozen + Klassifikationskopf**
   Hauptansatz der Arbeit. DINOv3 wird zuerst als eingefrorener Feature
   Extractor genutzt. Darauf wird ein einfacher Klassifikationskopf trainiert,
   um zu pruefen, ob self-supervised vortrainierte Repraesentationen fuer den
   Schleifgrad geeignet sind.

3. **DeiT-Tiny from scratch**
   Lizenzbewusste ViT-Kontrollarchitektur. Das Modell wird als Vision
   Transformer ohne externe vortrainierte Gewichte betrachtet und dient als
   ViT-from-scratch-Untergrenze, nicht als erwarteter Leistungsfavorit.

4. **DINOv3 patchbasiert**  
   Qualitative Zusatzanalyse fuer lokale Klassifikationskarten bzw. Heatmaps.
   Da keine pixelgenauen Segmentierungsannotationen vorliegen, wird keine echte
   semantische Segmentierung trainiert.

## Zentrale Reproduzierbarkeitsregeln

- Alle Experimente sollen ueber versionierte Config-Dateien in `configs/` steuerbar sein.
- Train/Val/Test-Splits muessen vor jeder Patch-Erzeugung auf Originalbild-, Bauteil- oder Aufnahmeebene erfolgen.
- Patches eines Originalbildes, Bauteils oder derselben Aufnahme duerfen nicht ueber verschiedene Splits verteilt werden.
- Das Testset darf nicht fuer Modellauswahl, Hyperparameter-Tuning, Schwellenwertwahl oder fruehe Designentscheidungen verwendet werden.
- Seeds, Config, Git-Commit, Paketversionen und relevante Hardwareinformationen muessen pro Experiment dokumentiert werden.
- Rohdaten, Checkpoints, Modellgewichte, Caches und grosse Outputs duerfen nicht in GitHub eingecheckt werden.

## Repository-Struktur

```text
configs/              # Versionierte Experimentkonfigurationen
configs/templates/    # Platzhalter-Configs, keine finalen Trainingsconfigs
data/                 # Datensatzdokumentation und ggf. Split-Manifeste, keine Rohdaten in Git
docs/                 # Methodische Dokumentation und Lizenznotizen
notebooks/            # Explorative Analysen, keine produktionskritische Logik
scripts/              # Hilfs- und Smoke-Test-Skripte, keine finalen Trainingslaeufe
src/
  data/               # Datenpruefung, Split- und Dataset-Logik
  models/             # Modellaufbau
  training/           # Trainingsschleifen
  evaluation/         # Metriken, Auswertung, Visualisierung
  utils/              # Gemeinsame Hilfsfunktionen
reports/
  figures/            # Kuratierte Abbildungen fuer Bericht/Thesis
  tables/             # Kuratierte Tabellen fuer Bericht/Thesis
outputs/              # Lokale Experimentoutputs, nicht fuer GitHub
```

## Naechste Schritte

1. Gruppierte Split-Strategie anhand der `q1`- bis `q4`-Gruppierung validieren.
2. Split-Manifest erzeugen, ohne Rohdaten oder sensible lokale Pfade zu committen.
3. Finale Hyperparameter experimentbezogen in Config-Dateien dokumentieren.
4. Erst danach kurze, reproduzierbare Trainingslaeufe starten.

Fuer die fruehe strukturelle Datensatzpruefung in Kapitel 3.1 steht
[docs/dataset_audit.md](docs/dataset_audit.md) zur Verfuegung.

Weitere Details stehen in [docs/project_context.md](docs/project_context.md),
[docs/model_selection.md](docs/model_selection.md),
[docs/training_comparison_design.md](docs/training_comparison_design.md),
[docs/experiment_plan.md](docs/experiment_plan.md) und
[data/README.md](data/README.md).
