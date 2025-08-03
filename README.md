# PhiASM
PhiASM

```
# PhiASM

**PhiASM** to nowoczesny, modularny interpreter i REPL dla języka niskiego poziomu nowej generacji.
Łączy zwięzłość i ekspresję makr z czytelnością oraz bezpieczeństwem typowym dla wyższych języków.
Idealny do nauki, eksperymentów oraz jako podstawa własnych języków blisko sprzętu.

---

## Najważniejsze funkcje

- **Modularna maszyna wirtualna** z dispatch table (łatwe dodawanie instrukcji)
- **Enum dla flag** (ZERO, NEG, CARRY) – czytelność i bezpieczeństwo
- **Bezpieczne makra** (rozszerzane regexem, nie psują kodu)
- **Obsługa HEX/BIN/DEC** w operandach (np. `MOV R1, 0xFF`)
- **Etykiety i skoki** (adresy rozwiązywane w parserze)
- **REPL z komendami na kropkę** (np. `.help`, `.registers`)
- **Debugowanie, breakpoints, krokowanie**
- **Przygotowanie pod GUI/IDE** (kolorowanie oddzielone od logiki)
- **Łatwe testowanie (input_func)**

---

## Instalacja

Wymagania:
- Python 3.7+
- (Opcjonalnie) `pyreadline3` dla lepszego REPL na Windows:  
  ```
  pip install pyreadline3
  ```

---

## Uruchomienie

**REPL (zalecane):**
```bash
python phiasm.py
```

**Uruchomienie programu z pliku:**
```bash
python phiasm.py twoj_program.asm
```

---

## Przykład kodu PhiASM

```asm
; Przykład: suma liczb od 1 do 10
MOV R1, 0      ; suma
MOV R2, 1      ; licznik

loop:
ADD R1, R2
ADD R2, 1
CMP R2, 11
JNZ loop, R2
OUT R1
END
```

---

## Komendy REPL (zawsze z kropką!)

| Komenda             | Opis                                      |
|---------------------|-------------------------------------------|
| `.help`             | Wyświetl pomoc                            |
| `.reset`            | Resetuj maszynę wirtualną                 |
| `.registers`        | Pokaż stan rejestrów                      |
| `.stack`            | Pokaż stan stosu                          |
| `.debug`            | Włącz/wyłącz tryb debug                   |
| `.step`             | Włącz/wyłącz tryb krok-po-kroku           |
| `.break <linia>`    | Ustaw breakpoint na linii                 |
| `.run <plik>`       | Uruchom kod z pliku                       |
| `.load <plik>`      | Załaduj i wyświetl kod z pliku            |
| `.macro <nazwa> <ciało>` | Zdefiniuj makro                      |
| `.quit` / `.exit`   | Wyjdź z REPL                              |

---

## Makra

Makra pozwalają na definiowanie własnych skrótów/instrukcji złożonych.

**Definicja makra:**
```
.macro sum MOV R1, R1 + R2
```
**Użycie w kodzie:**
```
@sum
```

---

## Obsługiwane instrukcje

- Arytmetyczne: `MOV`, `ADD`, `SUB`, `MUL`, `DIV`, `MOD`
- Logiczne: `AND`, `OR`, `XOR`, `NOT`
- Bitowe: `SHL`, `SHR`, `ROL`, `ROR`
- Kontrola przepływu: `JMP`, `JMPZ`/`JZ`, `JNZ`, `CALL`, `RET`
- Stos: `PUSH`, `POP`
- Porównania: `CMP`
- Wejście/wyjście: `IN`, `OUT`, `PRINT`
- Specjalne: `HALT`, `END`

---

## Debugowanie

- Tryb debug (`.debug`) pokazuje aktualną instrukcję, rejestry i flagi.
- Tryb krokowy (`.step`) pozwala wykonywać kod krok po kroku.
- Breakpointy (`.break <linia>`) zatrzymują wykonanie na wybranej linii.

---

## Przykładowa sesja REPL

```
PhiASM REPL v2.0
Wpisz '.help' aby zobaczyć dostępne komendy
PhiCode > .registers
Stan rejestrów:
   R0 = 0 (0x00000000)
   R1 = 0 (0x00000000)
   ...
PhiCode > MOV R1, 0x10
PhiCode > ADD R1, 5
PhiCode > .registers
Stan rejestrów:
   R0 = 0 (0x00000000)
   R1 = 21 (0x00000015)
   ...
PhiCode > .quit
Do widzenia!
```

---

## Własne rozszerzenia

Kod jest modularny i łatwo go rozbudować o nowe instrukcje, makra, wsparcie dla GUI/web IDE, 
animacje działania VM itd.

---

## Licencja

MIT

---

**Autor:**  
Projekt demonstracyjny nowoczesnego języka niskiego poziomu PhiASM.

```


