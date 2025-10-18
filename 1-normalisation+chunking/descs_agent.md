### Kontext und Aufgabe:
Wir verarbeiten Produkttexte für **medizinische Kühl- und Gefriergeräte** (z. B. Labortiefkühlschränke, Blutplasmagefrierschränke). Wir haben werbende Produktbeschreibungen die wie bereinigen und aufbessern müssen.

### Aufgabe

Analysiere den folgenden Produkttext und generiere eine **mehrabsätzige Beschreibung**, wobei **jeder Absatz genau eine Eigenschaft oder Merkmal** des Produkts behandelt.

### Regeln

1. **Extrahiere ALLE Informationen** aus dem Text – auch wenn sie verstreut sind.
2. **Strukturiere sie in logische Absätze** - je Aspekt.
3. **Erhalte alle technischen Daten 1:1** (keine Änderungen an Werten/Einheiten).
4. **Füge Hersteller/Produktname/Kategorie hinzu**, falls nicht im Text enthalten.
5. **Erkläre Fachbegriffe kurz**, wenn nötig (z. B. *"Vakuum-Isolierung: sorgt für Energieeffizienz"*).

### Sprache

- Die Geräte werden in **Kliniken, Laboren, Blutbanken und Apotheken** eingesetzt.
- Zielgruppe: **Fachpersonal** (Ärzte, Labormitarbeiter) **und** **Einkäufer** (keine Fachkenntnisse).
- Sachlich, keine Werbetext

---
### Eingabe:
{text}

---
### Ausgabeformat (JSON-Array):
```json
[
  "Absatz 1: Produktidentifikation + Hauptzweck (Hersteller, Produktname, Kategorie, Einsatzbereich)",
  "Absatz 2: Technische Daten (Maße, Gewicht, Volumen, Temperaturbereich)",
  "Absatz 3: Funktionen/Sonderausstattungen (Alarme, Steuerung, Zertifizierungen)",
  "Absatz 4: Einsatzbereiche/Zielgruppe (Kliniken, Labore, Blutbanken etc.)",
  "Absatz 5: Sicherheits- und Compliance-Features (DIN-Normen, ATEX etc.)"
]
```

### Wichtige Regeln
- Jeder Absatz **muss Hersteller, Produktbezeichnung und Zusätze eingearbeitet haben** so das jeder Absatz eindeutig einem Produkt zugeordnet werden kann
- Wenn Kategorie und andere Details vorliegen, die den Absatz kontextuell erweitern, füge das hinzu
