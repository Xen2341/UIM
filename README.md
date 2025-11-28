# Predikce výskytu diabetu

Tento projekt se zabývá predikcí výskytu diabetu na základě klinických parametrů žen.  
Model využívá klasifikační algoritmus **Random Forest**, doplněný o předzpracování dat, imputaci chybějících hodnot a standardizaci.  
Hodnocení modelu probíhá pomocí metriky **Matthews Correlation Coefficient (MCC)** a křížové validace (CV).

---

##  Cíl projektu
- Předpovědět přítomnost diabetu (0 = bez diabetu, 1 = diabetes) z klinických údajů.
- Ověřit výkonnost modelu na nezávislé testovací sadě.
- Udržet strukturu projektu přehlednou a reprodukovatelnou.

---

## Použitá data
Dataset obsahuje následující klinické příznaky:

| Název | Popis |
|-------|--------|
| Pregnancies | Počet těhotenství |
| Glucose | Hladina glukózy v plazmě po 2h OGTT |
| BloodPressure | Diastolický krevní tlak (mmHg) |
| SkinThickness | Tloušťka kožní řasy tricepsu (mm) |
| Insulin | Dvouhodinová hladina inzulinu (μU/ml) |
| BMI | Index tělesné hmotnosti |
| DiabetesPedigreeFunction | Genetická predispozice |
| Age | Věk |
| Outcome | Cílová proměnná (0 = bez diabetu, 1 = diabetes) |

---

## Struktura projektu
```
Project_BPC-UIM_SUIN-main/
│
├─ src/
│   ├─ config.py
│   ├─ schema.py
│   ├─ data_prep.py
│   ├─ models.py
│   ├─ feature_select.py
│   └─ train.py
│
├─ artifacts/
│   ├─ pipeline.pkl
│   ├─ feature_importance.csv
│   ├─ selected_features.json
│   └─ cv_report.json
│
├─ data/
│   └─diabetes_data.csv
├─ testing.py
├─ requirements.txt
└─ README.md
```

---

##  Průběh pipeline

1. **Kontrola dat (`schema.py`)**  
   Ověření přítomnosti všech potřebných sloupců.

2. **Předzpracování (`data_prep.py`)**  
   - Nahrazení nereálných nul hodnotami NaN.  
   - Imputace mediánem.  
   - Standardizace číselných proměnných.

3. **Trénink (`train.py`)**  
   - Křížová validace (`StratifiedKFold`, 5 foldů).  
   - Vyhodnocení pomocí MCC.  
   - Uložení nejlepšího modelu (full vs reduced).  

4. **Výběr příznaků (`feature_select.py`)**  
   - Získání `feature_importance_` z Random Forestu.  
   - Výběr TOP-K nejdůležitějších příznaků (defaultně 4).

5. **Predikce (`testing.py`)**  
   - Načtení uložené pipeline.  
   - Ověření vstupního CSV.  
   - Vytvoření výstupu `predictions.csv`.

---

## Použitý model
- **RandomForestClassifier**
  - `n_estimators = 400`
  - `min_samples_leaf = 2`
  - `class_weight = "balanced"`
  - `random_state = 42`

Model byl zvolen pro svou robustnost a možnost interpretace pomocí feature importance.

---

##  Výsledky

| Metrika | Výsledek                                    |
|----------|---------------------------------------------|
| Nejlepší MCC (CV) | **0.000**                                   |
| Nejvýznamnější příznaky | Glucose, BMI, Age, DiabetesPedigreeFunction |

Výsledky křížové validace (průměr a směrodatná odchylka) jsou uložené v `artifacts/cv_report.json`.

---
---

## Spuštění projektu
### 1. Instalace prostředí
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2. Trénink modelu
Spustí pipeline, validaci a uloží model:

```bash
python -m src.train data/diabetes_data.csv artifacts
```

### 3. Predikce na nových datech
```bash
python testing.py data/diabetes_data.csv
```

Výstup: predictions.csv
Pokud dataset obsahuje i Outcome, zobrazí se Self-check MCC a Confusion Matrix.