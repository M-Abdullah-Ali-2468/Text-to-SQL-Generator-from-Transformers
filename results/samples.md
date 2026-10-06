# Qualitative Dev Samples

## Five Correct Examples

### Correct Example 1

**Question:** Who is the player that wears number 42?

**Gold SQL:** `SELECT Player WHERE No. = '42'`

**Model SQL:** `SELECT Player WHERE No. = '42'`

**Result:** Correct

---

### Correct Example 2

**Question:** Which player who played for the Rockets for the years 1986-92?

**Gold SQL:** `SELECT Player WHERE Years for Rockets = '1986-92'`

**Model SQL:** `SELECT Player WHERE Years for Rockets = '1986-92'`

**Result:** Correct

---

### Correct Example 3

**Question:** What is the earliest years any of the incumbents were first elected? 

**Gold SQL:** `SELECT MIN(First elected)`

**Model SQL:** `SELECT MIN(First elected)`

**Result:** Correct

---

### Correct Example 4

**Question:** Who wrote the episode that received 1.83 million U.S. viewers?

**Gold SQL:** `SELECT Written by WHERE U.S. viewers (millions) = '1.83'`

**Model SQL:** `SELECT Written by WHERE U.S. viewers (millions) = '1.83'`

**Result:** Correct

---

### Correct Example 5

**Question:** How many tries against got the club with 62 tries for?

**Gold SQL:** `SELECT Tries against WHERE Tries for = '62'`

**Model SQL:** `SELECT Tries against WHERE Tries for = '62'`

**Result:** Correct

---

## Five Wrong Examples

### Wrong Example 1

**Question:** What position does the player who played for butler cc (ks) play?

**Gold SQL:** `SELECT Position WHERE School/Club Team = 'Butler CC (KS)'`

**Model SQL:** `SELECT Player WHERE School/Club Team = 'butler (ks)'`

**Failure:** Wrong column, Wrong value

---

### Wrong Example 2

**Question:** How many schools did player number 3 play at?

**Gold SQL:** `SELECT COUNT(School/Club Team) WHERE No. = '3'`

**Model SQL:** `SELECT COUNT(No.) WHERE Player = '3'`

**Failure:** Wrong column, Missing/extra condition column

---

### Wrong Example 3

**Question:** What school did player number 21 play for?

**Gold SQL:** `SELECT School/Club Team WHERE No. = '21'`

**Model SQL:** `SELECT Player WHERE No. = '21'`

**Failure:** Wrong column

---

### Wrong Example 4

**Question:** What player played guard for toronto in 1996-97?

**Gold SQL:** `SELECT Player WHERE Position = 'Guard' AND Years in Toronto = '1996-97'`

**Model SQL:** `SELECT Player WHERE Years in Toronto = '1996-97'`

**Failure:** Missing/extra condition column

---

### Wrong Example 5

**Question:** Who are all of the players on the Westchester High School club team?

**Gold SQL:** `SELECT Player WHERE School/Club Team = 'Westchester High School'`

**Model SQL:** `SELECT Player WHERE Nationality = 'westchester high school'`

**Failure:** Missing/extra condition column

---

