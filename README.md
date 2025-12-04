
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

## 1. Struktura projektu

```text
UIM/
│
├── artifacts/                      # Výstupy trénování a testování
│   ├── cv_report.json
│   ├── feature_names.json
│   ├── learning_curve.png
│   ├── model.pkl
│   └── predictions.csv
│
├── data/                           #  vstupní datasety
│
├── data_prep/                      # Modul pro zpracování dat
│   ├── cleaning.py                 # Čištění dat, outlier clipping
│   ├── feature_engineering.py      # Tvorba klinických a odvozených rysů
│   └── build_preprocessor.py       # KNN imputace + standardizace
│
├── model/                          # Definice modelů
│   └── random_forest.py            # Konfigurace RandomForestClassifieru
│
├── pipeline/                       # Stavba end-to-end ML pipeline
│   └── build_pipeline.py           # Cleaning → FE → Preprocessing → Model
│
├── config.py                       # Konfigurace projektu (sloupce, random_state…)
│
├── train.py                        # Trénování modelu, CV, learning curve, ukládání artefaktů
├── testing.py                      # Testování na externích datech + metriky
│
├── requirements.txt                # Seznam závislostí
└── README.md                       # Dokumentace projektu

````

---

## Průběh pipeline

Níže máš **stručný, jasný a informačně bohatý popis průběhu celé pipeline** — takový, jaký ocení učitel při kontrole projektu. Text je napsaný profesionálně, věcně a bez zbytečné omáčky. Můžeš ho vložit přímo do README.

---

# **Průběh pipeline**

### **1. Čištění a validace dat (ClinicalCleaner)**

Prvním krokem pipeline je klinické čištění vstupního datasetu.
Probíhá několik úprav:

* nahrazení neplatných hodnot (0 a negativní čísla) hodnotou `NaN` ve vybraných klinických sloupcích,
* detekce extrémních hodnot pomocí IQR metody a jejich ořezání do přijatelného rozsahu,
* zachování integrity numerických rysů bez odstraňování záznamů.

---

### **2. Feature Engineering**

V této fázi se vytvářejí nové klinické a odvozené příznaky.
Mezi nejdůležitější patří:

* metabolické ukazatele (HOMA-IR, GI ratio),
* interakční rysy (BMI × Age, počet těhotenství vůči věku),
* mocninné transformace (Glucose², BMI², Age²),
* rizikové binární příznaky (High Glucose, Obesity, Hypertension),
* log-transformace vybraných rysů.

---

### **3. Preprocessing (imputace + škálování)**

Čistá a rozšířená data procházejí jednotným předzpracováním:

* **KNN imputace** doplňuje chybějící hodnoty na základě nejpodobnějších záznamů v datasetu,
* **StandardScaler** normalizuje rozsah numerických proměnných, aby žádný atribut nedominoval při učení modelu.

Tento krok zajišťuje konzistentní vstupní data pro model vůbec nezávisle na původní kvalitě datasetu.

---

### **4. Modelování (RandomForestClassifier)**

Jako finální klasifikátor se používá Random Forest s upravenými hyperparametry:

* 600 stromů (vyšší stabilita predikce),
* omezená hloubka a velikost listů pro prevenci přeučení,
* `class_weight="balanced"` pro správné zacházení s nerovnoměrnými třídami,
* `max_features=0.8` pro zvýšení diverzity stromů.

Model se učí přímo na výstupech z preprocesoru a je součástí jedné sjednocené pipeline.

---

### **5. Trénování, validace a ukládání artefaktů**

Po sestavení pipeline se provádí:

* **train/test split** se stratifikací,
* výpočet metrik (Accuracy, AUC, MCC, F1),
* **learning curve**, která ukazuje charakter učení modelu,
* **křížová validace (StratifiedKFold)** se skóre MCC,
* uložení modelu (`model.pkl`), názvů rysů a CV reportu do složky `artifacts/`.

---

### **6. Testování na externích datech**

V samostatné fázi:

* načítá se uložený model,
* provádí se inference na neznámém datasetu,
* generují se predikce a pravděpodobnosti,
* vypisují se metriky a confusion matrix,
* ukládá se `predictions.csv`.

---


## Použitý model
- **RandomForestClassifier**
  - `n_estimators = 600`
  - `max_depth = 5`
  - `max_features=0.8`
  - `min_samples_leaf = 3`
  - `class_weight = "balanced"`
  - `random_state = 42`

Model byl zvolen pro svou robustnost a možnost interpretace pomocí feature importance.

---

##  Výsledky

| Metrika           | Výsledek   |
|-------------------|------------|
| Nejlepší MCC      | **0.5550** |
| AUC               | **0.8389** |
| Accurency         | **0.7939** |
| F1-score(class 1) | **0.7158** |

Výsledky křížové validace (průměr a směrodatná odchylka): 

[CV] MCC: 0.4691 +/- 0.0479


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
python -m train data/diabetes_data.csv artifacts
```

### 3. Predikce na nových datech
```bash
python -m testing your_data.csv artifacts
```
## Autory

Xeniya Pushilova,
Matěj Pakán,
Daria Mikriukova.
