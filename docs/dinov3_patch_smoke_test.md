# DINOv3 Patch-Smoke-Test

## Zweck

Dieser Test prueft nur, ob die technische Pipeline fuer eine patchbasierte
DINOv3-Feature-Extraktion funktioniert:

1. ein einzelnes Bild laden oder ein synthetisches Beispielbild erzeugen,
2. das Bild in nicht ueberlappende Patches zerlegen,
3. jeden Patch mit einem eingefrorenen DINOv3-Backbone in ein Embedding
   ueberfuehren,
4. einfache Diagnostik und eine Patch-Aehnlichkeitsvisualisierung in
   `outputs/` speichern.

Der Test ist kein Training, keine Modellauswahl und keine Bewertung des
Datensatzes. Er nutzt keine Labels und kein Testset. Er darf deshalb vor der
finalen Split-Strategie nur als Machbarkeits- und Infrastrukturtest
interpretiert werden.

## Methodische Grenze

Patchbasierte Schleifgradkarten duerfen spaeter erst auf bereits gesplitteten
Originalbildern, Bauteilen oder Aufnahmen erzeugt werden. Patches aus demselben
Originalbild duerfen nicht ueber Train, Validation und Test verteilt werden.

Die in diesem Smoke-Test erzeugte Patch-Aehnlichkeitskarte ist keine
Segmentierung und keine Schleifgradvorhersage. Sie visualisiert nur, ob
DINOv3 fuer die ausgeschnittenen Patches numerische Features liefern kann und
ob diese Features ueber die Bildpositionen zurueckprojiziert werden koennen.

## Modell- und Lizenzhinweis

Die Standard-Config nutzt `facebook/dinov3-vits16-pretrain-lvd1689m`, die
kleinste offizielle DINOv3-ViT-Variante auf Hugging Face. Die Modellkarte ist
als `dinov3-license` markiert und erfordert vor dem Download ggf. eine
Zustimmung zu den Zugriffsbedingungen. Gewichte und Caches duerfen nicht ins
Repository eingecheckt werden.

## Ausfuehrung

Alle Befehle werden aus dem Repository-Root ausgefuehrt:

```powershell
cd C:\path\to\Bachelorarbeit
```

Eine passende Minimalumgebung kann mit der pinbaren Requirements-Datei
installiert werden:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dinov3-prototype.txt
```

```powershell
.\.venv\Scripts\python.exe scripts\dinov3_patch_smoke_test.py --config configs/dinov3_patch_smoke_test.yaml
```

Optional mit einem echten Einzelbild:

```powershell
.\.venv\Scripts\python.exe scripts\dinov3_patch_smoke_test.py --config configs/dinov3_patch_smoke_test.yaml --image C:\path\to\image.jpg
```

Ein trockener Pipeline-Test ohne DINOv3-Download:

```powershell
.\.venv\Scripts\python.exe scripts\dinov3_patch_smoke_test.py --config configs/dinov3_patch_smoke_test.yaml --dry-run
```

## Erwartete Outputs

Alle Outputs liegen unter `outputs/dinov3_patch_smoke_test/` und bleiben durch
`.gitignore` lokal:

- `summary.json`: Config, Git-Commit, Paketversionen, Patchkoordinaten,
  Embedding-Diagnostik oder Dry-Run-Hinweis.
- `synthetic_brushed_surface.png`: nur bei synthetischem Fallback.
- `patch_similarity_overlay.png`: einfache Overlay-Visualisierung der
  Patch-Aehnlichkeit zum mittleren Patch-Embedding, falls DINOv3 gelaufen ist.
