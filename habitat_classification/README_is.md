# Vistgerðaflokkun

Flokkaðu íslenskar gervihnattamyndir í 71 vistgerð.
## TIl að byrja

```bash
# Setja upp
pip install -r requirements.txt

# Ræsa þjón
python api.py
```

Þjónninn keyrir á `http://localhost:4321`.

## Verkefnið

Gefin er 35×35 punkta mynd með 15 rásum (Sentinel-2 litrófsrásir + landslagsupplýsingar). Spáðu hvaða vistgerð myndin sýnir (ein af 71).

### Inntak

Numpy fylki með lögun `(15, 35, 35)`, gildi á bilinu 0-1.

| Rás | Lýsing |
|-----|--------|
| 0-11 | Sentinel-2 litrófsrásir |
| 12 | Hæð yfir sjávarmáli |
| 13 | Halli |
| 14 | Stefna |

### Úttak

Heiltala frá 0 til 70 sem táknar vistgerð.

## Gögn

Þjálfunargögn eru í `data/`:

- `train/patches.npy` - 5186 myndir
- `train.csv` - Merki

```python
from utils import load_training_data

patches, labels = load_training_data()
```

## API Endpoint

```
POST /predict
{
    "patch": "<base64 kóðað numpy fylki>"
}

Svar:
{
    "prediction": 42
}
```

## Einkunnagjöf

**Mælikvarði**: Vegið F1 skor

| Árangur | Skor |
|---------|------|
| Slembigrunnlíkan | ~4% |
| Gott | 20-25% |
| Frábært | 30%+ |

## Ábendingar

1. Notaðu allar 15 rásir
2. Notaðu litrófsvísitölur (NDVI, NDWI)
3. Meðhöndlaðu ójafnvægi í flokkum

Gangi þér vel!
