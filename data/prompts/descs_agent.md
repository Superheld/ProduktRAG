### Kontext und Aufgabe:
Wir verarbeiten Produkttexte für **medizinische Kühl- und Gefriergeräte** (z. B. Labortiefkühlschränke, Blutplasmagefrierschränke). Wir haben werbende Produktbeschreibungen die wie bereinigen und aufbessern müssen.

### Aufgabe

Analysiere den folgenden Produkttext und führe folgende Aufgaben aus:

1. **Extrahiere die Produktkategorie** aus folgenden vordefinierten Optionen:
   - "Laborkühlschrank"
   - "Medikamentenkühlschrank"
   - "Blutkühlschrank"
   - "Gefrierschrank"
   - "Ultratiefkühlschrank"
   - "Labortiefkühlschrank"
   - "Sonstiges"

2. **Generiere eine mehrabsätzige Beschreibung**, wobei **jeder Output-Absatz einem Original-Absatz aus dem Input-Text entspricht**. Jeder Absatz sollte **50-100 Wörter** umfassen.

**Anzahl der Absätze:** 3-5 Absätze, je nach vorhandenen Absätzen im Input-Text. **Ein Input-Absatz = Ein Output-Absatz.** Erstelle nur Absätze, für die ausreichend Informationen vorliegen.

### Regeln

1. **Extrahiere ALLE Informationen** aus dem Text – auch wenn sie verstreut sind.
2. **Strukturiere sie in logische Absätze** - je Aspekt.
3. **Erhalte alle technischen Daten 1:1** (keine Änderungen an Werten/Einheiten).
4. **Bereinige den Produktnamen** um SEO-Erweiterungen wie Kategorien oder technischen Daten (z.B. "Laborkühlschrank", "(+5Grad)", ...)
5. **Füge Hersteller, bereinigten Produktnamen hinzu**, falls nicht im Text enthalten.
6. **Erkläre Fachbegriffe kurz**, wenn nötig (z. B. *"Vakuum-Isolierung: sorgt für Energieeffizienz"*).
7. **Optimiere für Findbarkeit:**
   - Nutze Synonyme und alternative Begriffe (z.B. "Kühlschrank", "Kühlgerät", "Kühlaggregat")
   - Gib praktische Kontexte (z.B. "für Medikamentenlagerung" statt nur "Temperaturbereich 2-8°C")
   - Verwende Fach- UND Alltagssprache
8. **NUR vorhandene Daten nutzen - KEINE Halluzination:**
   - Erfinde keine technischen Daten, Features oder Eigenschaften
   - Nutze ausschließlich Informationen aus dem Input-Text
   - Keine Vermutungen oder allgemeines Wissen ergänzen
9. **Produktnamen korrekt verwenden**
   - Beim Nennen des Produktnamens: Verwende NUR Hersteller + Modellbezeichnung
   - Die Kategorie (z.B. "Laborkühlschrank") kann im Text erscheinen, aber NIEMALS als Teil des Produktnamens selbst
   - Alternative: Nutze neutrale Begriffe wie "Das Gerät", "Der Kühlschrank", "Das Modell"
10. **Kategorie extrahieren:**
   - Nutze **NUR die vorgegebenen Kategorien** aus der Liste
   - Basierend auf Produkttyp im Titel und/oder Beschreibung
   - Bei Unsicherheit oder fehlenden Informationen: "Sonstiges"
   - Beispiele: "Laborkühlschrank" (aus Titel), "Medikamentenkühlschrank" (aus Beschreibung)

11. **Bereinigten Produktnamen als Title extrahieren:**
   - Extrahiere Hersteller + Modellbezeichnung aus dem Input-Text
   - Entferne SEO-Erweiterungen (Kategorien, technische Daten in Klammern)
   - ✅ RICHTIG: "[Hersteller] [Modellbezeichnung]"
   - ❌ FALSCH: "[Hersteller] [Modellbezeichnung] [Kategorie]"
   - ❌ FALSCH: "[Hersteller] [Modellbezeichnung] (+X°C bis +Y°C)"

### Sprache

- Die Geräte werden in **Kliniken, Laboren, Blutbanken und Apotheken** eingesetzt.
- Zielgruppe: **Fachpersonal** (Ärzte, Labormitarbeiter) **und** **Einkäufer** (keine Fachkenntnisse).
- Sachlich, keine Werbetext

---
## Beispiele für gute und schlechte Absätze

### ✅ GUTER Absatz (ca. 70 Wörter):
> "Der [Hersteller] [Modell] bietet einen Nutzinhalt (Kühlvolumen, Fassungsvermögen, Lagerkapazität) von [X] Litern. Damit eignet sich das Kühlgerät für mittlere bis große Laboranforderungen und kann mehrere Medikamentenkisten oder Probenbehälter gleichzeitig lagern. Die Innenausstattung umfasst [Y] höhenverstellbare Ablagen (Roste, Einlegeböden), die eine flexible Raumaufteilung für unterschiedliche Behältergrößen ermöglichen."

**Warum gut:**
- Produktname OHNE Kategorie-Suffix ("[Hersteller] [Modell]", NICHT "[Hersteller] [Modell] [Kategorie]")
- Synonyme in Klammern ("Kühlvolumen, Fassungsvermögen")
- Praktischer Kontext ("für Medikamentenkisten")
- Technische Daten 1:1 übernommen
- Sachlich, keine Werbesprache

### ❌ SCHLECHTER Absatz:
> "Der revolutionäre Laborkühlschrank bietet großzügigen Stauraum. Mit modernster Technologie ausgestattet, ist er die beste Wahl für anspruchsvolle Labore. Die hochwertige Verarbeitung garantiert jahrelange Zuverlässigkeit."

**Warum schlecht:**
- Werbesprache ("revolutionär", "beste Wahl", "modernste Technologie") ❌
- Kein Hersteller, kein Modellname ❌
- Keine konkreten technischen Daten ❌
- Halluzination ("hochwertige Verarbeitung" stand nicht im Text) ❌
- Keine Synonyme, keine Findbarkeit ❌

---
### Eingabe:
{text}

---
### Ausgabeformat (JSON-Objekt):
Gib ein JSON-Objekt zurück mit der extrahierten Kategorie und den generierten Absätzen:

```json
{
  "title": "Hersteller Modellbezeichnung (OHNE Kategorie, OHNE technische Daten)",
  "category": "Kategorie aus der vorgegebenen Liste",
  "descriptions": [
    "Absatz 1: Produktidentifikation + Hauptzweck (Hersteller, Produktname, Einsatzbereich)",
    "Absatz 2: Technische Daten (Maße, Gewicht, Volumen, Temperaturbereich)",
    "Absatz 3: Funktionen/Sonderausstattungen (Alarme, Steuerung, Zertifizierungen)",
    "Absatz 4: Einsatzbereiche/Zielgruppe (Kliniken, Labore, Blutbanken etc.) - OPTIONAL",
    "Absatz 5: Sicherheits- und Compliance-Features (DIN-Normen, ATEX etc.) - OPTIONAL"
  ]
}
```

**Wichtig:**
- Die Kategorie muss **exakt** aus der vorgegebenen Liste stammen
- Erstelle **3-5 Absätze**, je nach verfügbaren Informationen im Input-Text
- Nicht alle 5 Absätze sind Pflicht - nur die, für die ausreichend Daten vorliegen

### Wichtige Regeln
- **Ein Input-Absatz = Ein Output-Absatz** - Behalte die Absatz-Struktur des Original-Texts bei
- Jeder Absatz **muss Hersteller + bereinigte Produktbezeichnung eingearbeitet haben** so das jeder Absatz eindeutig einem Produkt zugeordnet werden kann
- **Produktname = NUR Hersteller + Modell** (OHNE Kategorie-Suffix)
- Die Kategorie kann separat im Text erscheinen (z.B. "...ist ein Laborkühlschrank, der...")
- Produktnamen müssen **von SEO-Erweiterungen bereinigt** werden (Kategorie-Begriffe wie "Laborkühlschrank", "Gefrierschrank" etc. aus dem Namen entfernen)
- **KEINE Halluzination:**
  - Erfinde NIEMALS technische Daten, Features oder Eigenschaften
  - Nutze AUSSCHLIESSLICH Informationen aus dem Input-Text
  - Keine Vermutungen, keine Annahmen, kein allgemeines Wissen
  - Beispiel FALSCH: "Die hochwertige Edelstahlverkleidung..." (wenn nicht im Text) ❌
  - Beispiel RICHTIG: "Das Gerät bietet einen Nutzinhalt von 280 Litern..." (aus Text übernommen) ✓
- **Findbarkeit optimieren:**
  - Synonyme in Klammern ergänzen (z.B. "Nutzinhalt (Kühlvolumen, Fassungsvermögen)")
  - Praktische Kontexte geben (z.B. "für Medikamentenlagerung")
  - Fach- UND Alltagssprache verwenden
