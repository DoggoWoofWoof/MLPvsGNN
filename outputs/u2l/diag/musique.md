# U2l diagnostic: musique / s1sel (decides nothing)

Pick `kt_cos:ada_f12_k0@0.85`. Ledger (U2e's, reproduced): `{"TAU_CUT": 22, "BUDGET": 8, "UNREACHED": 32}`, 56 questions missing a gold.

Questions: `{"questions_missing_a_gold": 56, "by_causes": {"BUDGET": 8, "TAU_CUT": 19, "TAU_CUT+UNREACHED": 2, "UNREACHED": 27}, "every_missed_gold_in_a_split_list": 0, "ALL_now": 0.9635, "ALL_if_the_split_lists_were_whole": 0.9635}`

## By cause

### HELD

```json
{
 "golds": 101,
 "distance": {
  "0": 83,
  "1": 13,
  "2": 5,
  "3": 0,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 56,
  "ner": 61,
  "knn": 66
 },
 "name_in_question": 22,
 "name_match_B": 38,
 "kinds": {
  "(none)": 101
 }
}
```

### TAU_CUT

```json
{
 "golds": 22,
 "distance": {
  "0": 0,
  "1": 15,
  "2": 7,
  "3": 0,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 11,
  "ner": 5,
  "knn": 0
 },
 "name_in_question": 3,
 "name_match_B": 0,
 "kinds": {
  "(none)": 22
 }
}
```

### BUDGET

```json
{
 "golds": 8,
 "distance": {
  "0": 0,
  "1": 0,
  "2": 8,
  "3": 0,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 0,
  "ner": 0,
  "knn": 0
 },
 "name_in_question": 0,
 "name_match_B": 0,
 "kinds": {
  "(none)": 8
 }
}
```

### UNREACHED

```json
{
 "golds": 32,
 "distance": {
  "0": 0,
  "1": 0,
  "2": 27,
  "3": 5,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 0,
  "ner": 0,
  "knn": 0
 },
 "name_in_question": 4,
 "name_match_B": 1,
 "kinds": {
  "(none)": 32
 }
}
```

## Links against NER

```json
{
 "w2s_pairs": 2343920,
 "ner_pairs": 800278,
 "both": 76337,
 "share_of_w2s_in_ner": 0.0326,
 "share_of_ner_in_w2s": 0.0954,
 "golds_one_link_from_B": {
  "in B": 3140,
  "knn": 30,
  "ner": 152,
  "ner+knn": 59,
  "none": 185,
  "structural": 199,
  "structural+knn": 35,
  "structural+ner": 123,
  "structural+ner+knn": 141
 }
}
```

