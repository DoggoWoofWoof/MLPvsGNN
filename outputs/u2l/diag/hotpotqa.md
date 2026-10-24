# U2l diagnostic: hotpotqa / s1sel (decides nothing)

Pick `kt_cos:ada_f12_k0@0.85`. Ledger (U2e's, reproduced): `{"TAU_CUT": 13, "BUDGET": 36, "UNREACHED": 20}`, 67 questions missing a gold.

Questions: `{"questions_missing_a_gold": 67, "by_causes": {"BUDGET": 35, "BUDGET+UNREACHED": 1, "TAU_CUT": 13, "UNREACHED": 18}, "every_missed_gold_in_a_split_list": 0, "ALL_now": 0.9556, "ALL_if_the_split_lists_were_whole": 0.9556}`

## By cause

### HELD

```json
{
 "golds": 65,
 "distance": {
  "0": 56,
  "1": 4,
  "2": 5,
  "3": 0,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 28
 },
 "name_in_question": 29,
 "name_match_B": 16,
 "kinds": {
  "(none)": 65
 }
}
```

### TAU_CUT

```json
{
 "golds": 13,
 "distance": {
  "0": 0,
  "1": 4,
  "2": 9,
  "3": 0,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 4
 },
 "name_in_question": 1,
 "name_match_B": 0,
 "kinds": {
  "(none)": 13
 }
}
```

### BUDGET

```json
{
 "golds": 36,
 "distance": {
  "0": 0,
  "1": 4,
  "2": 31,
  "3": 1,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 4
 },
 "name_in_question": 0,
 "name_match_B": 0,
 "kinds": {
  "(none)": 36
 }
}
```

### UNREACHED

```json
{
 "golds": 20,
 "distance": {
  "0": 0,
  "1": 0,
  "2": 16,
  "3": 1,
  "4": 3,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 0
 },
 "name_in_question": 1,
 "name_match_B": 1,
 "kinds": {
  "(none)": 20
 }
}
```

