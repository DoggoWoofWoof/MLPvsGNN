# U2l diagnostic: squad / s1sel (decides nothing)

Pick `kt_cos:ada_f12_k0@0.85`. Ledger (U2e's, reproduced): `{"TAU_CUT": 0, "BUDGET": 30, "UNREACHED": 1}`, 31 questions missing a gold.

Questions: `{"questions_missing_a_gold": 31, "by_causes": {"BUDGET": 30, "UNREACHED": 1}, "every_missed_gold_in_a_split_list": 0, "ALL_now": 0.9793, "ALL_if_the_split_lists_were_whole": 0.9793}`

## By cause

### BUDGET

```json
{
 "golds": 30,
 "distance": {
  "0": 0,
  "1": 14,
  "2": 11,
  "3": 5,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 14
 },
 "name_in_question": 0,
 "name_match_B": 10,
 "kinds": {
  "(none)": 30
 }
}
```

### UNREACHED

```json
{
 "golds": 1,
 "distance": {
  "0": 0,
  "1": 0,
  "2": 0,
  "3": 1,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 0
 },
 "name_in_question": 0,
 "name_match_B": 0,
 "kinds": {
  "(none)": 1
 }
}
```

